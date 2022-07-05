import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from baseline.pytorch import TensorDef, BaseLayer
from baseline.model import register_model, create_model_for
from baseline.train import register_trainer, register_training_func, create_trainer, EpochReportingTrainer
from baseline.utils import save_vocabs, get_model_file, get_metric_cmp
from eight_mile.pytorch.layers import EmbeddingsStack
from baseline.embeddings import create_embeddings_reduction
from mead.tasks import Task, Backend, read_config_file_or_json, index_by_label, print_dataset_info, register_task
from eight_mile.downloads import DataDownloader
from eight_mile.pytorch.layers import SequenceLoss
from baseline.reader import SeqPredictReader, register_reader, _filter_vocab, _all_predefined_vocabs
from torch.utils.data import DataLoader
from eight_mile.utils import Offsets, listify, revlut, to_spans, per_entity_f1, span_f1, conlleval_output
from eight_mile.progress import create_progress_bar
from eight_mile.pytorch.optz import OptimizerManager
import sys
import logging
from collections import Counter
import json

TASK_NAME = 'meta_tagger'
logger = logging.getLogger(TASK_NAME)


@register_task
class FewShotTaggerTask(Task):
    """Define a new Task in mead for meta-learning approach to tagging

    A Task is a whole new problem that can have multiple backends, trainers, readers, and models.
    Most Tasks are built in to the framework, but it is possible to create your own by inheriting
    this class and passing in `--task_modules <path_to_task_addon.py>`

    This particular task is going to be implemented only in PyTorch for now.  It is created to
    do few-shot learning by using support data (an entire few-shot set), and a query set used
    to test (and possibly train the dataset).  This setup, while suitable for training and inference,
    is not really efficient for test time, as the support data is passed into the forward() function
    and re-encoded every time.

    """

    def __init__(self, mead_settings_config, **kwargs):
        super().__init__(mead_settings_config, **kwargs)
        logger.debug("Adding task [%s]", TASK_NAME)

    @classmethod
    def task_name(cls):
        """Implementations will use this string name to register new handlers

        :return:
        """
        return TASK_NAME

    def _create_backend(self, **kwargs):
        """Boilerplate code to set up the backend information"""
        backend = Backend(self.config_params.get('backend', 'pytorch'), kwargs)
        backend.load(self.task_name())

        return backend

    def _setup_task(self, **kwargs):
        """Some readers support functions to 'clean' sentences, dont allow"""
        super()._setup_task(**kwargs)
        self.config_params.setdefault('preproc', {})
        self.config_params['preproc']['clean_fn'] = None

    def initialize(self, embeddings):
        """Download any necessary embeddings, create embedding and dataset indices, and embeddings object

        :param embeddings:  The embeddings index (usually embeddings.{json,yml}
        """
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print_dataset_info(self.dataset)
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        vocab_sources = [self.dataset['train_file'], self.dataset['valid_file']]
        if 'test_file' in self.dataset:
            vocab_sources.append(self.dataset['test_file'])

        vocabs = self.reader.build_vocab(vocab_sources, min_f=Task._get_min_f(self.config_params), **self.dataset)
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs,
                                                                   self.config_params['features'])
        save_vocabs(self.get_basedir(), self.feat2index)

    def _get_features(self):
        """Define the features as the set of embeddings to init model, this what we will want for most configurations

        For our tagger specifically, this is usually a pretrained Transformer with the label layer removed

        This will get passed into the model at creation time
        :return: The features
        """
        return self.embeddings

    def _get_labels(self):
        """Define the labels as a list of the processed labels from the reader

        This will get passed into the model at creation time
        :return:
        """
        return self.reader.label2index

    def _reorganize_params(self):
        """This is boilerplate configuration code that is used to allow backwards compat config files

        :return:
        """
        train_params = self.config_params['train']
        train_params['batchsz'] = train_params['batchsz'] if 'batchsz' in train_params else self.config_params[
            'batchsz']
        train_params['test_batchsz'] = train_params.get('test_batchsz', self.config_params.get('test_batchsz', 1))
        unif = self.config_params.get('unif', 0.1)
        model = self.config_params['model']
        model['unif'] = model.get('unif', unif)
        lengths_key = model.get('lengths_key', self.primary_key)
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key
        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v

    def _load_dataset(self):
        """Boilerplate dataset loading"""
        read = self.config_params['reader'] if 'reader' in self.config_params else self.config_params['loader']
        sort_key = read.get('sort_key')
        bsz, vbsz, tbsz = Task._get_batchsz(self.config_params)
        self.train_data = self.reader.load(
            self.dataset['train_file'],
            self.feat2index,
            bsz,
            shuffle=True,
            sort_key=sort_key,
        )
        self.valid_data = self.reader.load(
            self.dataset['valid_file'],
            self.feat2index,
            vbsz,
        )
        self.test_data = None
        if 'test_file' in self.dataset:
            self.test_data = self.reader.load(
                self.dataset['test_file'],
                self.feat2index,
                tbsz,
            )


def metalearn_collator_fn(data: List[Dict]):
    """PyTorch function to form batches on the fly

    This implementation is simplified in that it only supports batch sizes of 1.  There is still an episode
    per batch, but we dont allow multiple per batch yet...  On my 2080ti its

    TODO: there are lot of lengths being passed here, only y_lengths being used right now.  Could use to improve efficiency
    possibly, or they can just be removed.  Right now we just use Offsets.PAD values to mask out invalid tokens

    :param data: A list of elements
    :return: A mini-batch
    """
    if len(data) > 1:
        raise Exception("We currently only support physical batch size of 1, use `grad_accum` to change eff batch size")
    d = data[0]
    support = torch.stack(d['support'])
    support_labels = torch.stack(d['support_labels'])
    support_lengths = torch.tensor(d['support_lengths'])
    query = torch.stack(d['query'])
    query_lengths = torch.tensor(d['query_lengths'])
    y = torch.stack(d['y'])
    y_lengths = torch.tensor(d['y_lengths'])
    return {
        'support': support,
        'support_labels': support_labels,
        'support_lengths': support_lengths,
        'query': query,
        'query_lengths': query_lengths,
        'y': y,
        'y_lengths': y_lengths
    }


@register_reader(task=TASK_NAME, name='default')
class FewNERDPreprocReader(SeqPredictReader):
    """A Reader for the FewNERD preprocessed file format

    This class overrides the MEAD SeqPredictReader (in baseline/reader.py) which is an abstract reader for handling
    tagger data.  That class usually handles most of the difficulties of packing the data and building the
    labels, but because this file format is very different, we end up having to override a few more functions
    than usual

    To get these files, run FewNERD's download script with the episode-data
    The file format includes paired {support,query} data and labels as a JSONL format
    We will consume each of these and form a Dataset-like object that can be consumed by the DataLoader

    TODO: you could cache the Tensors with a torch.save() or saving as an NPY/NPZ which would speed up the loader
    drastically on subsequent runs.  Then you can just check the presence of a cache file and use the
    tensor dataset from there

    """

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, num_workers=8, samples_per_epoch=sys.maxsize, **kwargs):
        """This function gets called by create_reader() and it passes the reader/loader block in

        :param vectorizers: A list of vectorizers, one per feature (currently we support one feature)
        :param trim: Whether to trim batches to a minimum length, its not really used at the moment
        :param truncate: Whether to remove the last batch if its not an even batch size (this doesnt matter ATM)
        :param mxlen: The maximum length (ignored)
        :param num_workers: The number of worker threads to read the data
        :param samples_per_epoch Optional arg to limit the size of an epoch using a random sampler
        :param kwargs: Ignored
        """
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        if len(self.vectorizers) > 1:
            raise Exception("We currently only support a single word vectorizer")
        self.num_workers = num_workers
        self.truncate = truncate
        self.samples_per_epoch = samples_per_epoch

    def build_vocab(self, files, **kwargs):
        """Unfortunately, as the format is substantially different from usual NER, we have to subclass this

        The version here is simplified from the base class.  The main change is we will need to read the label
        set in from a specific field in the JSON and we cannot use the `label_vectorizer.count()` as this field
        is separate from the token stream.  For vectorizers with predefined vocab lists (anything with a subword vocab)
        there is not much to do in this file (_all_predefined_vocabs() checks that for us), except to build the label
        vocab.  That is the brunt of the work here

        :param files: The files to use to build the label list
        :param kwargs:
        :return:
        """
        if _all_predefined_vocabs(self.vectorizers):
            vocabs = {k: v.vocab for k, v in self.vectorizers.items()}
            have_vocabs = True
        else:
            have_vocabs = False
            vocabs = {k: Counter() for k in self.vectorizers.keys()}

        pre_vocabs = None
        labels = Counter()

        for file in files:
            if file is None:
                continue
            # Read the entire JSONL file
            examples = self.read_examples(file)
            # For each example, there is a field called 'types' which includes everything but O
            for example in examples:
                labels += Counter(example['types'] + ['O'])
                if not have_vocabs:
                    for k, vectorizer in self.vectorizers.items():
                        vocab_example = vectorizer.count(example)
                        vocabs[k].update(vocab_example)

        # This is unusual, but if you arent using a pretrained vocab, we used the vectorizers to count the
        # features, so we can drop features below min_f occurrences.  We wont do anything for pretrained vocabs
        if not have_vocabs:
            vocabs = _filter_vocab(vocabs, kwargs.get('min_f', {}))
        # We offset the label starting at some magic values
        base_offset = len(self.label2index)
        # This shouldnt happen one this dataset, copied from base class out of caution
        labels.pop(Offsets.VALUES[Offsets.PAD], None)
        for i, k in enumerate(labels.keys()):
            self.label2index[k] = i + base_offset
        if not have_vocabs and pre_vocabs:
            vocabs = pre_vocabs
        return vocabs

    def read_examples(self, tsfile):
        """This is an abstract function in the base class, here we just slurp the JSONL

        :param tsfile: A JSONL file to read from (e.g., train_5_5.jsonl)
        :return: A list of the JSON-ified samples
        """
        with open(tsfile) as rf:
            samples = [json.loads(line.strip()) for line in rf]

        return samples

    def load(self, filename, vocabs, batchsz=1, shuffle=False, sort_key=None):
        """Override the loader to return a DataLoader, since we dont really care about TensorFlow

        The behavior here is similar to the base class, but simplified to produce a DataLoader.
        This also will make the trainer function a bit simpler

        :param filename: The file to load (e.g. train_5_5.json)
        :param vocabs: The vocabs (Dict[str, int])
        :param batchsz: The batch size (currently we support 1, due to collation function limitations)
        :param shuffle: Should we randomly shuffle the data
        :param sort_key: Should we sort the data (ignored for now)
        :return:
        """
        texts = self.read_examples(filename)
        dataset = self.convert_to_tensors(texts, vocabs)
        # Allow random sampling in any dataset this is shuffled (usually this is train only)
        if self.samples_per_epoch < len(dataset) and shuffle:
            logger.info("Setting epoch size to [%d] samples for [%s]", self.samples_per_epoch, filename)
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=self.samples_per_epoch)
            # sampler option is mutually exclusive with shuffle
            shuffle = False
        else:
            sampler = None
        loader = DataLoader(dataset, batchsz, shuffle=shuffle, sampler=sampler,
                            num_workers=self.num_workers, pin_memory=True, drop_last=self.truncate,
                            collate_fn=metalearn_collator_fn)

        return loader

    def convert_to_tensors(self, texts, vocabs):
        """The texts coming in are already processed into a dictionary

        This class converts all of the text data to tensors and forms a "map-style dataset"
        :param texts: Texts come in as a JSON object, with a sub-object for query and support
        :param vocabs: The vocabs for vectorizers.  This is a dict, but we assume only one vectorizer for now
        :return: Return the vectorized data
        """
        ts = []
        word_vocab = list(vocabs.values())[0]
        word_vectorizer = list(self.vectorizers.values())[0]
        for i, sample in enumerate(texts):
            example = {}
            query = sample['query']
            support = sample['support']

            example['support'] = []
            example['support_labels'] = []
            example['support_lengths'] = []
            for sentence, labels in zip(support['word'], support['label']):
                pretok = [{'text': w, 'labels': l} for w, l in zip(sentence, labels)]
                tokens, len_s = word_vectorizer.run(pretok, word_vocab)
                support_labels, len_y = self.label_vectorizer.run(pretok, self.label2index)
                if len_y != len_s:
                    raise Exception(f"Unexpected length mismatch in support: {len_y} vs {len_s}")
                example['support'].append(torch.from_numpy(tokens))
                example['support_labels'].append(torch.from_numpy(support_labels))
                example['support_lengths'].append(len_y)

            example['query'] = []
            example['y'] = []
            example['query_lengths'] = []
            for sentence, labels in zip(query['word'], query['label']):
                pretok = [{'text': w, 'labels': l} for w, l in zip(sentence, labels)]
                tokens, len_s = word_vectorizer.run(pretok, word_vocab)
                support_labels, len_y = self.label_vectorizer.run(pretok, self.label2index)
                if len_y != len_s:
                    raise Exception(f"Unexpected length mismatch in query: {len_y} vs {len_s}")
                example['query'].append(torch.from_numpy(tokens))
                example['y'].append(torch.from_numpy(support_labels))
                example['query_lengths'].append(len_y)

            example['y_lengths'] = example['query_lengths']

            # example['ids'] = sample['index']

            ts.append(example)

        return ts


class FewShotTaggerModel(nn.Module):
    """Abstract base class for few-shot tagger with minimal definition

    The forward() function is presumably used in training when we need access to the support and query vectors
    to meta-learn.  At test time, we dont want this.  We could make forward() smarter (and then slower), or we can
    just define a new function that takes in the pre-encoded support (which is what is done here)

    To make this work with ONNX you would need a wrapper nn.Module with a forward() function to call the predict().
    See `Embedder` defined in mead/pytorch/exporters.py
    """
    def __init__(self):
        super().__init__()

    task_name = TASK_NAME

    def save(self, outname: str):
        """Save out the model
        :param outname: The name of the checkpoint to write
        :return:
        """
        torch.save(self, outname)

    def create_layers(self, embeddings, **kwargs):
        """Create underlying model layers

        :param embeddings:
        :param kwargs:
        :return:
        """
    def forward(self, support, label_support, query) -> Tuple[torch.Tensor, torch.Tensor]:
        """Meta-learning training signature. Not for use in eval

        :param support: A one-hot vector of the support words
        :param label_support: A one-hot vector of the support labels
        :param query: A one-hot vector of the query
        :return: A
        """

    def predict(self, support_enc, label_support, query):
        """Prediction signature, the support is encoded to a 3D tensor by this point

        :param support_enc:
        :param label_support:
        :param query:
        :return:
        """

@register_model(TASK_NAME, 'default')
class NNShotModel(FewShotTaggerModel):
    """Implementation suitable for meta-learning

    This is an implementation of NNShot for meta-learning where we learn from episodes of (suppport, query)
    data.  When we actually move this model into production, we have to avoid using our forward() function as
    we have defined it here, since we will not want to encode the support each time.

    """

    def __init__(self):
        super().__init__()

    def save(self, outname: str):
        """Save out the model
        :param outname: The name of the checkpoint to write
        :return:
        """
        torch.save(self, outname)

    @classmethod
    def create(cls, embeddings, labels, **kwargs) -> 'NNShotModel':
        model = cls()
        # model.feature_key = kwargs.get('lengths_key', 'word_lengths').replace('_lengths', '')
        model.pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.create_layers(embeddings, **kwargs)
        return model

    def create_layers(self, embeddings, **kwargs):
        # The head is removed for NN-shot, so all we need is this portion
        self.embeddings = self.init_embed(embeddings, **kwargs)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', kwargs.get('embed_reduction_type', 'concat'))
        reduction = create_embeddings_reduction(embed_reduction_type=reduction, **kwargs)
        embeddings_dropout = float(kwargs.get('embeddings_dropout', self.pdrop))
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction)

    def distance(self, x, y):
        x = nn.functional.normalize(x)
        y = nn.functional.normalize(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        logits = -((x - y) ** 2).sum(dim=2)
        return logits

    def decode(self, scores, label_support, query):
        # (T_q, 1) & (1, T_k)
        # (T_q, T_k)
        query = query.view(-1)

        # Mask out padded values
        mask = (query != Offsets.PAD).unsqueeze(1) & (label_support != Offsets.PAD).unsqueeze(0)
        scores = scores.masked_fill(mask == False, -1e9)
        best = scores.argmax(1)

        nearest = torch.full((query.shape[0], len(self.labels)), -1e9, device=scores.device)
        # TODO: this is not efficient, replace with a gather?
        for label in label_support.unique():
            label_mask = label_support == label
            nearest[:, label] = torch.max(scores[:, label_mask], 1)[0]
        return label_support[best], nearest

    def forward(self, support, label_support, query) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: change this to look more like the usual input, right now we have to make sure the key matches the file
        support_enc = self.embeddings({'word': support})
        query_enc = self.embeddings({'word': query})
        H = query_enc.shape[-1]
        query_enc = query_enc.view(-1, H)
        support_enc = support_enc.view(-1, H)
        scores = self.distance(query_enc, support_enc)
        best, pred = self.decode(scores, label_support.view(-1), query)
        return best.view(query.shape), pred.view(query.shape + (-1,))

    def predict(self, support_enc, label_support, query):
        with torch.no_grad():
            query_enc = self.embeddings({'word': query})
            H = query_enc.shape[-1]
            query_enc = query_enc.view(-1, H)
            support_enc = support_enc.view(-1, H)
            scores = self.distance(support_enc, query_enc)
            best, pred = self.decode(scores, label_support)
            return best.view(query.shape), pred.view(query.shape + (-1,))

    def create_loss(self, **kwargs):
        """Create the loss function.

        :param kwargs:
        :return:
        """
        return SequenceLoss(LossFn=nn.CrossEntropyLoss)


@register_training_func(TASK_NAME)
def fit(model_params, ts, vs, es, **kwargs):
    """The few-shot tagger approach is pretty much boilerplate

    The only differences are the model name and a simplification for the fact that a PyTorch DataLoader is guaranteed

    :param model_params:
    :param ts:
    :param vs:
    :param es:
    :param kwargs:
    :return:
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file(TASK_NAME, 'pytorch', kwargs.get('basedir'))
    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(trainer.model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics


@register_trainer(task=TASK_NAME, name='default')
class FewShotTaggerTrainerPyTorch(EpochReportingTrainer):
    """Trainer based on typical tagger trainer

    """

    def __init__(self, model, **kwargs):
        super().__init__()
        if type(model) is dict:
            checkpoint = kwargs.get('checkpoint')
            if checkpoint:
                model['checkpoint'] = checkpoint
            model = create_model_for(TASK_NAME, **model)
        # Setting the grad accum will give you a larger effective batch size. The effective batch size will be
        # batchsz * grad_accum
        # This is important in this current implementation because the batch is limited to the episodes
        self.grad_accum = int(kwargs.get('grad_accum', 1))
        logger.info("Gradient Accumulation steps [%d]", self.grad_accum)
        self.gpus = int(kwargs.get('gpus', 1))
        # By default support IOB1/IOB2
        self.span_type = kwargs.get('span_type', 'iob')
        self.verbose = kwargs.get('verbose', False)

        logger.info('Setting span type %s', self.span_type)
        self.model = model
        self.loss = model.create_loss()
        self.idx2label = revlut(self.model.labels)
        self.clip = float(kwargs.get('clip', 5))
        self.optimizer = OptimizerManager(self.model, **kwargs)
        if self.gpus > 1:
            logger.info("Trainer for PyTorch meta tagger currently doesnt support multiple GPUs.  Setting to 1")
            self.gpus = 1
        if self.gpus > 0 and self.model.gpu:
            self.model = model.cuda()
            self.loss = self.loss.cuda()
        else:
            logger.warning("Requested training on CPU.  This will be slow.")
        # This determines if we do stat logging in the middle of an epoch (its prob a good idea to do it for this model)
        self.nsteps = kwargs.get('nsteps', sys.maxsize)

    def save(self, model_file):
        self.model.save(model_file)

    @staticmethod
    def _get_batchsz(batch_dict):
        return batch_dict['y'].shape[0]

    def make_input(self, example):
        """Convert the input example dictionary into CUDA tuple to input to model

        :param example:
        :return:
        """
        support = example['support']
        query = example['query']
        support_labels = example['support_labels']
        y_lengths = example['y_lengths']
        y = example['y']
        if self.model.gpu:
            support = support.cuda()
            query = query.cuda()
            support_labels = support_labels.cuda()
            y = y.cuda()
            y_lengths = y_lengths.cuda()
        return support, support_labels, query, y, y_lengths

    def process_output(self, guess, truth, sentence_lengths):

        # For acc
        correct_labels = 0
        total_labels = 0
        truth_n = truth.cpu().numpy()
        # For f1
        gold_chunks = []
        pred_chunks = []

        # For each sentence
        for b in range(len(guess)):
            sentence = guess[b]
            if isinstance(sentence, torch.Tensor):
                sentence = sentence.cpu().numpy()
            sentence_length = sentence_lengths[b]
            gold = truth_n[b, :sentence_length]
            sentence = sentence[:sentence_length]

            valid_guess = sentence[gold != Offsets.PAD]
            valid_gold = gold[gold != Offsets.PAD]
            valid_sentence_length = np.sum(gold != Offsets.PAD)
            correct_labels += np.sum(np.equal(valid_guess, valid_gold))
            total_labels += valid_sentence_length
            # TODO: check if there are any issues with the calculation in this function when we dont have B and I
            # If there is an issue, we can fix it with a custom `to_spans` to convert to B-<x> I-<x> etc.
            gold_chunks.append(set(to_spans(valid_gold, self.idx2label, self.span_type, self.verbose)))
            pred_chunks.append(set(to_spans(valid_guess, self.idx2label, self.span_type, self.verbose)))

        return correct_labels, total_labels, gold_chunks, pred_chunks

    def _test(self, ts, **kwargs):

        self.model.eval()
        total_sum = 0
        total_correct = 0

        gold_spans = []
        pred_spans = []

        metrics = {}
        steps = len(ts)
        pg = create_progress_bar(steps)
        total_loss = 0.
        for batch_dict in pg(ts):
            support, support_labels, query, y, lengths = self.make_input(batch_dict)
            best, predictions = self.model(support, support_labels, query)
            loss = self.loss(predictions, y)
            total_loss += loss.item()
            correct, count, golds, guesses = self.process_output(best, y, lengths)
            total_correct += correct
            total_sum += count
            gold_spans.extend(golds)
            pred_spans.extend(guesses)

        total_acc = total_correct / float(total_sum)
        avg_loss = total_loss / float(total_sum)
        metrics['acc'] = total_acc
        metrics['avg_loss'] = avg_loss
        metrics['f1'] = span_f1(gold_spans, pred_spans)
        if self.verbose:
            # TODO: Add programmatic access to these metrics?
            conll_metrics = per_entity_f1(gold_spans, pred_spans)
            conll_metrics['acc'] = total_acc * 100
            conll_metrics['tokens'] = total_sum.item()
            logger.info(conlleval_output(conll_metrics))
        return metrics

    def _train(self, ts, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_norm = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        self.optimizer.zero_grad()

        for i, batch_dict in enumerate(pg(ts)):

            support, support_labels, query, y, lengths = self.make_input(batch_dict)
            _, predictions = self.model(support, support_labels, query)
            # Divide the loss by the grad accum so it stays the same as batch size increases
            loss = self.loss(predictions, y) / self.grad_accum
            loss.backward()

            # We only do optimizer steps on gradient accum steps
            if (i + 1) % self.grad_accum == 0 or (i + 1) == steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            # TODO: not sure if we should do this calc, each "batch" is different size
            bsz = self._get_batchsz(batch_dict)
            report_loss = loss.item() * bsz
            epoch_loss += report_loss
            epoch_norm += bsz
            self.nstep_agg += report_loss
            self.nstep_div += bsz
            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                metrics['lr'] = self.optimizer.current_lr
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_norm)
        metrics['lr'] = self.optimizer.current_lr

        return metrics
