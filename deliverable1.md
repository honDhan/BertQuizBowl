# CMSC 470 Project Deliverable 1

Anu Basil, Dhanvee Ivaturi, Ge Huang, Rishi Nangia, Sam Miller

## Things that are working

We’ve decided that we’ll be using BERT for our guesser. We also considered using Word2Vec, Doc2Vec, and GloVe to generate word embeddings, and then do something with those. From our research, this seemed less effective than using BERT, and would involve more work since we’d still need to turn the embeddings into guesses.

Getting BERT to actually work is proving to be harder than expected. We were able to get an almost functional guesser using BERT. It can take qanta data as input and it can train with no errors.

We were able to find a couple potential datasets that we will try out on our QA system. The links are below. These might be helpful when testing out our system later.

https://quac.ai/

https://github.com/facebookresearch/ELI5

## Things that aren’t working

Getting BERT to a usable point, however, is a far more daunting task. Currently, it is very slow and consumes a lot of computing resources. We also haven't figured out how to get it to use features. Further (and likely because of this), the accuracy is 0 (run on a small subset of the data). 

## Timeline Changes

By this time, we wanted to finalize our dataset and submit a barebones submission to gradescope. We definitely aren’t ready for that, and we are shifting that goal to within the next 2 weeks. However, we are definitely making solid progress towards project completion.
