SHELL:=/bin/bash
PATH:=$(PATH):/home/gchrupala/repos/recurrent:/home/gchrupala/repos/text-rep/src:/home/gchrupala/repos/GroundHog/experiments/nmt/preprocess:/home/gchrupala/repos/GroundHog/experiments/nmt

HOG=/home/gchrupala/repos/GroundHog/experiments/nmt
LM=/home/gchrupala/repos/recurrent/data/twitter_sample_2_elmanpy-4.params.pkl

VOCABULARY=/home/gchrupala/repos/text-rep/data/american-english
OOV_TYPE=anno
TESTSET=5 6 7 8 9
DEVSET=0 1 2 3 4

all: dev-0.wer

dev.corpus.tweet.utf8: /home/gchrupala/repos/text-rep/data/lexnorm/corpus.tweet
	cat /home/gchrupala/repos/text-rep/data/lexnorm/corpus.tweet | iconv -f latin1 -t utf8 > $@


lexnorm.ori lexnorm.tgt lexnorm.oov: dev.corpus.tweet.utf8
	cat dev.corpus.tweet.utf8 \
            |   lexnorm_parse.py word-wise $(OOV_TYPE) lexnorm.ori lexnorm.tgt lexnorm.oov $(VOCABULARY) 

train-%.ori dev-%.ori: lexnorm.ori
	crossval.py lexnorm.ori 10 .ori

train-%.tgt dev-%.tgt: lexnorm.tgt
	crossval.py lexnorm.tgt 10 .tgt


train-%.ori.vocab.pkl train-%.ori.pkl: train-%.ori
	preprocess.py -d train-$*.ori.vocab.pkl -v 128 -b train-$*.ori.pkl -p train-$*.ori -t

train-%.ori.ivocab.pkl: train-%.ori.vocab.pkl
	invert-dict.py train-$*.ori.vocab.pkl train-$*.ori.ivocab.pkl

train-%.ori.h5: train-%.ori.pkl
	convert-pkl2hdf5.py  train-$*.ori.pkl  train-$*.ori.h5

train-%.tgt.vocab.pkl train-%.tgt.pkl: train-%.tgt
	preprocess.py -d train-$*.tgt.vocab.pkl -v 128 -b train-$*.tgt.pkl -p train-$*.tgt -t

train-%.tgt.ivocab.pkl: train-%.tgt.vocab.pkl
	invert-dict.py train-$*.tgt.vocab.pkl train-$*.tgt.ivocab.pkl

train-%.tgt.h5: train-%.tgt.pkl
	convert-pkl2hdf5.py  train-$*.tgt.pkl  train-$*.tgt.h5

train-%.ori.shuf.h5 train-%.tgt.shuf.h5: train-%.ori.h5 train-%.tgt.h5
	shuffle-hdf5.py  train-$*.ori.h5 train-$*.tgt.h5 \
                         train-$*.ori.shuf.h5 train-$*.tgt.shuf.h5

search_norm_model.npz: train-0.ori.shuf.h5 train-0.tgt.shuf.h5 \
                       train-0.ori.vocab.pkl train-0.tgt.vocab.pkl \
                 train-0.ori.ivocab.pkl train-0.tgt.ivocab.pkl 
	OMP_NUM_THREADS=1 train.py --proto=prototype_search_state --state $(HOG)/normalization.py 

%.oritgt: %.ori %.tgt
	paste $*.ori $*.tgt  > $@

model-%.pkl: $(LM) train-%.oritgt dev-%.oritgt
	OMP_NUM_THREADS=1 normalize.py --train=train-$*.oritgt --valid=dev-$*.oritgt --model_file=$@ \
                     --interval=1000 --lm=$(LM) \
                     --rate=0.05 --decay=0.1 --momentum=0.2 --hidden_size=400 --iterations=30 > $@.log


dev-%.wer: model-%.pkl dev-%.oritgt
	OMP_NUM_THREADS=1 normalize.py --model_file=model-$*.pkl --eval_mode < dev-$*.oritgt > $@

.PRECIOUS: model-%.pkl train-%.ori.pkl train-%.tgt.pkl train-%.ori train-%.tgt train-%.ori.vocab.pkl train-%.tgt.vocab.pkl \
           train-%.ori.ivocab.pkl train-%.tgt.ivocab.pkl train-%.ori.h5 train-%.tgt.h5 search_norm_model.npz

