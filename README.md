# LyricsGeneration
In this task, we trained a neural net to generate lyrics based on the provided melody.

During the training of the model, we had access to the lyrics of a song and its melody. The melodies are stored in .mid (MIDI files) and contain various types of information – notes, the instruments used, etc. we experiment with different methods to incorporate this information with the lyrics.

During the test phase, we generate lyrics for a provided melody automatically, based on the first input word.



Overview

In this assignment, we were asked to design and build an RNN that receives a melody sequence and a word and predicts the next word in a song. While working on this task, we faced different challenges, such as choosing optimal melody and word representation, picking the exacted melody part that represents the word, setting the RNN correctly, and preparing the test data to produce a good sequence generation. In this work, we describe our development process and our implementation considerations. We also present our suggested solutions, review the experiments’ results and evaluate the generated outputs. 

	Textual data
We used GENSIM, a framework for fast vector space and NLTK package for text processing and tokenizing.
We used pre-trained word embeddings in machine learning that Google provides – a news corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors).
We performed some general text processing on the Lyrics data to bring each song’s lyrics to a Python’s list of tokens. We used those tokens to build a vocabulary. Using the pretrained model we produced embedding matrix- where each word in the lyrics list has got its own matching word embedded vector, represented by 300 numbers. We will use this word embedded vector as the input to our model in case of Lyrics only, and as partial features for the other models
Some of the methods for text processing that we used are: lower text, tokenizing and remove punctuation, removing word that contain numbers, removing empty tokens, removing special characters. We also used syllables counting, for an optional implementation.

	Melody Extraction
As mentioned in the assignment requirements, we used PrettyMIDI package to handle and work with midi files. We tested and explored its abilities and limitations, and eventually realized that some work and further processing are needed, in order to extract the meaningful musical parts.
Our main concern was: How can we pick the best musical part which represents a leading role, that can be matched to the lyric’s words?
We know that MIDI file contains multiple channels, each channel is performed as a musical instrument, or vocal role. We know that some of the instruments are usually support the main melody by producing the songs’ rhythm and beats – such as drums, bass, background sounds, etc. They might be important to the songs’ harmony, but they are too general and has low contribution to the song’s words and singing part. For example, we can find similar repetitive drums part, appearing on different popular songs.
On the other hand, there are the melody or lead vocal channel, which can represent the singing part perfectly. This part is characterized by a high variability of intervals, low repetition, a variety of notes and notes durations, and sharp transitions between notes are pretty rare. The melody part also tells us how to sing the lyrics, since each musical note occurrence can be matched to a syllable of a word. In optimal situation, we could have taken the melody part, find its lyrics, spread the words sequence by dividing them to syllables and match the notes for each word!

In between these two edges, we can find variety of instruments and musical roles, each has its own contribution to the song’s harmony. So how can we choose among this range of instruments, which is the best melody representation?
We realized that a smart guess is needed here. we wanted to be able to pick a channel that is also a good-enough lyrics representation - hopefully represents the melody, based on heuristics and approaches that are commonly used for music analysis.
We followed some papers that deal with the task of melody extraction in MIDI files. They characterized the melodic structure and suggest different approaches to extract it, based on specific parameters and calculations. This paper [1], suggests 3 parameters that are calculated for each notes sequence, combines calculation results to a final formula and produce a ranking to the specific track.
These parameters are:
	Note Rate: The number of notes per unit time (Note rate) denotes the amount
	Level Crossing Rate (LCR):  which measures the “waviness” of a note sequence and tells about the rate of variation of a sequence. Sometimes, the genuine notes may simply move up and down (at a high frequency) but may not contribute to melody perception. For melody perception, the note sequence must be slow in pitch variation. 
	Distinct Note Count: Number of distinct notes per unit time. Tracks / channels containing less than 5 distinct notes in 10 sec duration are discarded.
This paper [2], highlights using of pitches entropy in order to recognize the melody sequence. In addition to these articles, we calculated and examined other parameters such as notes (pitches) and intervals varieties, total duration of songs, chords indicator metric, average notes amount per song and others.
During our experiments, we tried several approaches, calculations and combinations in order to identify the leading melody channel. We used MIDI files that tagged their own melody and vocal parts, to check if our metric chose correctly.
We produced the final formula for scoring and ranking the MIDI channels:
〖Score〗_inst=  1/2  ∙[(0.3∙NR+0.2∙Dist+0.2∙〖Entr〗_notes+0.3∙〖Entr〗_dur )-(CI-1)∙2]+1/2∙MI
NR is Notes Rate and Dist is Distinct Note Count, as suggested on [1].  Entr is notes (pitches) entropy as suggested in [2] and notes duration entropy. We found out that these parameters can be good predictors for leading channel. CI is chord indicator: a chord is occurrence of multiple notes that are played on the same time. Obviously, a human voice cannot produce chords, so if we find chords on our examined channel, we will reduce its score and prefer other options. 
MI is melody indicator, which increases the score significantly if a tagged melody channel was found. More specifically, it looks for the tags ["voc", "melody", "lead", "voice", "word"] in the instrument (channel) name. we noticed that many songs tagged their leading singer as main melody channel, so we realized that melody extraction can be very tricky if it based on MIDI tagging.
Our results for a little sample set of MIDI files were pretty satisfied, and we could recognize meaningful channels in the given MIDI files. However, especially when the melody leading part was not very significant, or when there was more than one vocal part - the melody part did not get the highest score. it could have been ranked second or even 3rd.  So, we decided to pick the top 3 highest channels, and we could assume that each of them can represent the melody part pretty well.
Note: in order to reduce training phase duration, during our experiments we used only the 1st ranked extracted melody, based on our calculation. However, our implementation supports extracting up to three leading channels.


	 Lyrics and Melody Matching

3.1	Melody part - Representation I – partial melody vector
After finding the probably-melody-channels, we extracted their notes from instrument notes list and transferred each of them into a sequence of notes (pitches). We normalized this sequence by dividing it in 128 - max pitch value - and set each of them as partial melody vector.
At this point we raised another concern: how do we want to shape the RNN input?  Given one word and a melody, how can we assign the best melody part to a specific word? A simple approach was to take the whole encoded melody vector and combine it to each of the words from the songs’ lyrics. We will examine this approach later on our second melody representation. However, this approach has some limitations: it uses large and unnecessary amount of data, and the concept of representing each word by the same whole melody sounds too general for us.
So, we tried to develop a method that “cuts” the relevant sequence part from the whole melody and attaches it to each specific word. To do so, we calculated estimated word position in the melody, based on its position in the lyrics. We set an interval of 20 notes before estimated word position and 20 notes after estimated word position and assumed that this 40-notes melody part probably catches the word actual location in the melody.
Eventually, we got the first 300 cells word embedded vector and 40 cells melody vector. We combined them to one lyrics melody vector with size of 340, that represent a word and an estimated melody part.

3.2	Melody part - Representation II – piano roll melody vector
In this representation we created a vector based on a sparse matrix, as produced by get-piano roll Pretty MIDI utility function. We used the same logic as presented above for instrument/channel selection, but this time, instead of creating a long list of notes - we produced piano roll matrix. In this representation we tried to catch the exact notes with respective to their timestamp occurrence, and we also marked the intervals (the silent parts, when no note is played) as unique integer.
In more details, we loop over piano roll matrix to find the pitch with the highest velocity value, for each timestep. Then, we add this pitch value to the vector. If no velocity value is found for a specific timestep – we assume to discover a “silent note” and mark it as 128.
After we finish looping over the melody, we get a complete vector with a fixed length of 1000 cells - trimmed or padded as needed, as a representation of the pitches that appear on the melody. Eventually, we normalized the vector and kept if for further processing. 
The next step was pretty similar to the creation of lyrics melody vector, as described above. We combined word embedded vector that has size of 300 with the piano roll melody vector and produced a long lyrics melody vector with length of 1300 (300 per word representation and 1000 per melody). At this melody representation, we did not consider word position - we attached the same piano roll melody vector to each word in the songs’ lyrics, which is a simple and less complex approach for word-melody representation.

	Target variables and Next Word Prediction

4.1	Target representation
For each input word and melody – whether melody is partial or complete - our target variable is the next word on lyrics, transformed to integers by tokenizer’s text_to_sequence function, and represented by one hot vector. We created the target words sequence at the same time we built the input data structure, so we could be “aware” to the word correct sequence and keep the words order.

4.2	Predict next word
When using the model for word sequence generation, we sorted the predicted words and used np.random.choice to avoid deterministic word selection. In this way, we not relay on argmax function for target selection - and the model is able to produce different word each time.
 
	RNN Implementation
5.1	Model 1: Simple LSTM
We used Keras Sequential model, which contains 3 LSTM layers, a dense layer in between and final dense layer that has vocabulary size, before activation. During our first experiments, our model tended to overfit; loss was reduced nicely over training data, but validation’s loss was high, stable and did not reduce.
We tried to avoid overfitting by generalizing  the model and using dropouts, adjusting neurons number in each LSTM layer, as well as limiting layers’ number. however, loss was still not reduced as expected for validation set.
We also performed some preprocessing methods on training data, such as normalization and scaling on input data – but no different was shown. At this point, we decided to look for improvement not from input data perspective – but from model’s optimizer, which has major influence on model’s ability to learn. While researching, we noticed that LSTMs for words generation usually use adaptive optimization algorithms, and we gave it a try.
 After testing popular adaptive optimizers, we found out that AdaDelta achieves the best results. AdaDelta [3] is an adaptive algorithm that requires minimum tuning and adjustment- and luckily, it helped our model to learn well, making loss decreasing both on training and validation datasets.
The final model has the following configuration:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #  =================================================================
lstm (LSTM)                  (None, 1, 256)            1594368  _________________________________________________________________
dropout (Dropout)            (None, 1, 256)            0        _________________________________________________________________
lstm_1 (LSTM)                (None, 1, 256)            525312   _________________________________________________________________
dropout_1 (Dropout)          (None, 1, 256)            0        _________________________________________________________________
lstm_2 (LSTM)                (None, 256)               525312   _________________________________________________________________
dense (Dense)                (None, 512)               131584
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         _________________________________________________________________
dense_1 (Dense)              (None, 7314)              3752082 _________________________________________________________________
activation (Activation)      (None, 7314)              0         =================================================================
Total params: 6,528,658
Trainable params: 6,528,658
Non-trainable params: 0
We set the batch size to 64. We tried 20 - 40 epochs, and notices that converge starts after ~20 epochs. we set validation split to 20% of entire training set.

	Model 2: Bi-directional LSTM
We were curious to see whether bidirectional LSTM will produce better outputs, comparing to unidirectional LSTM. We know that unidirectional LSTM processes and preserves past information – the data flows only forward. On the other hand, bidirectional LSTM can run forward and backward, and preserves information from both past and future. As a result, bidirectional can understand context better.
We set a limit of 20 epochs per training phase, since we did not recognize significant improvement or loss reduction afterwards.
The bidirectional LSTM model has the following configuration:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   =================================================================
lstm (LSTM)                  (None, 1, 256)            611328    _________________________________________________________________
dropout (Dropout)            (None, 1, 256)            0         _________________________________________________________________
bidirectional (Bidirectional (None, 1, 512)            1050624   _________________________________________________________________
lstm_2 (LSTM)                (None, 256)               787456    _________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         _________________________________________________________________
dense (Dense)                (None, 256)               65792     _________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         _________________________________________________________________
dense_1 (Dense)              (None, 7314)              1879698   =================================================================
Total params: 4,394,898
Trainable params: 4,394,898
Non-trainable params: 0  
	Results
Training and validation loss graphs are shown below, as appears on TensorBoard extension.
6.1	Simple LSTM
 
  
Figure 1: loss of "partial melody" simple LSTM model
  
Figure 2: loss of "piano roll melody" simple LSTM model

6.2	Bidirectional LSTM
  			 
Figure 3: loss of "partial melody" Bidirectional LSTM		Figure 4: loss of "piano roll melody" Bidirectional LSTM

	Output evaluation
Full outputs are provided in the last part of the report.
We performed lyrics generation, based on 3 different words, for 5 test songs, for two different RNN configurations and for two different melody presentations: partial melody and piano roll melody. We examined the output and we noticed that the models use variety of words. However, the produces sentences were not very clear or coherent. Among the two representations, the piano roll melody representation showed slightly better sentences: we could recognize some weak connections between sentences parts, such as verbs and nouns. So far, we did not recognize significant difference between unidirectional and bidirectional LSTM outputs.
