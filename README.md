# Rhythm Search Engine

## Description
A search engine that retrieves music having phrase boundaries similar to Mozart's K448 in terms of changes in rhythm and note density. 

## Demo

* `Music Retrieval Based on Rhythmic Similarity.ipynb` explains how the retrieval system finds the 10 pieces from MAESTRO dataset that is most similar to Mozart's K448 using.
* `Periodicity Feature Extraction.ipynb`: Rhythmic periodicity feature extraction from audio.
* `Dynamic Periodicity Warping.ipynb`: A distance metric that used in music retrieval.


### Todo
- [] `Evaluation.ipynb`: Evaluate and compare different methods on the MIREX dataset which contains manually annotated rhythmic pattern from 4 pieces of music.
- [] `Rhythmic retrieval from MIDI.ipynb`: 


## Result

* Top 20 similar pieces: `./Top20_pieces.csv`
* Top 10 similar segment: `./Top10_segments.csv`

Run `trim_audio.sh` get the segments.