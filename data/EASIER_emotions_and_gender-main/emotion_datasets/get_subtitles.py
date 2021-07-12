import os, sys
import argparse
import json
import pandas as pd
import datetime
from laserembeddings import Laser
import numpy as np
 
def read_json(json_path):
    ## TODO
    ## EmoryNLP: 
    #{
    #"season_id": "dev",
    #"episodes": [
        #{
        #"episode_id": "s01_e15",
        #"scenes": [
            #{
            #"scene_id": "s01_e15_c01",
            #"utterances": [
                #{
                #"utterance_id": "s01_e15_c01_u001",
                #"speakers": ["Rachel Green"],
                #"transcript": "Coffee.",
                #"tokens": [
                    #["Coffee", "."]
                #],
                #"emotion": "Neutral"
                #},
                #...
            #}
        #}
      #}
    return en_json

def get_most_similar(en_embed, de_embeds):
    en_embeds = np.repeat(en_embed, de_embeds.shape[0], axis=0)
    cos_sim = np.sum( # from laser laserembeddings/tests/test_laser.py
                en_embeds * de_embeds,
                axis=1) / (np.linalg.norm(en_embeds, axis=1) *
                           np.linalg.norm(de_embeds, axis=1))
    best_i =np.argmax(cos_sim)
    #print(cos_sim)
    return best_i, cos_sim[best_i] 

def get_time_in_s(time):
    #delta = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)
    delta = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
    return delta
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--en-csv', type=str, metavar='PATH', help='path to en corpus (csv)')
    parser.add_argument('--en-json', type=str, metavar='PATH', help='path to en subtitles (json)')
    parser.add_argument('--de-csv', type=str, metavar='PATH', help='path to german subtitles (csv)')
    parser.add_argument('--season', type=int,  help='season of srt')
    parser.add_argument('--episode', type=int,  help='episode of srt')
    parser.add_argument('--log-missing', type=str, metavar='PATH', help='where to log missing subtitles')
    args = parser.parse_args()

    if args.en_csv:
        en_csv = pd.read_csv(args.en_csv)
    if args.en_json:
        with open(json_path) as json_file:
            en_json = json.load(json_file)
    de_csv = pd.read_csv(args.de_csv)
    laser = Laser()

    #print(en_csv.index[1])
    with open(args.log_missing, 'w') as logfile:
        print("Sr No.,Utterance_de,Utterance_en,Candidates_de,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime,laser")
        for index, row in en_csv.iterrows():
            start_en = get_time_in_s(datetime.datetime.strptime(row.StartTime, '%H:%M:%S,%f'))
            end_en = get_time_in_s(datetime.datetime.strptime(row.EndTime, '%H:%M:%S,%f'))
            season = row.Season
            episode = row.Episode
            speaker = row['Speaker']
            emotion = row['Emotion']
            sentiment = row['Sentiment']
            en_utterance = row['Utterance']
            score = 0.0
            de_sentences =[]
            if not isinstance(en_utterance, str):
                        en_utterance = en_utterance.values[0]

            ## get the german subtitle
            de_row = de_csv.loc[(de_csv['Season'] == season) & (de_csv['Episode'] == episode) & (de_csv['StartTime'] == str(start_en)) & (de_csv['EndTime'] == str(end_en)) ]
            if len(de_row) ==0:
                ## get +/- 5s texts and compare with laser
                de_rows_minus = de_csv.loc[(de_csv['Season'] == season) & 
                                           (de_csv['Episode'] == episode) &
                                           ( (de_csv['StartTime'] == str(start_en)) |
                                             (de_csv['StartTime'] == str(start_en - datetime.timedelta(seconds=1))) | 
                                             (de_csv['StartTime'] == str(start_en - datetime.timedelta(seconds=2))) |
                                             (de_csv['StartTime'] == str(start_en - datetime.timedelta(seconds=3))) |
                                             (de_csv['StartTime'] == str(start_en - datetime.timedelta(seconds=4))) |
                                             (de_csv['StartTime'] == str(start_en - datetime.timedelta(seconds=5)))
                                            ) ]
                de_rows_plus = de_csv.loc[(de_csv['Season'] == season) & 
                                           (de_csv['Episode'] == episode) &
                                           ( (de_csv['StartTime'] == str(start_en + datetime.timedelta(seconds=1))) | 
                                             (de_csv['StartTime'] == str(start_en + datetime.timedelta(seconds=2))) |
                                             (de_csv['StartTime'] == str(start_en + datetime.timedelta(seconds=3))) |
                                             (de_csv['StartTime'] == str(start_en + datetime.timedelta(seconds=4))) |
                                             (de_csv['StartTime'] == str(start_en + datetime.timedelta(seconds=5)))
                                           ) ]
                de_rows = [de_rows_minus , de_rows_plus]
                de_rows = pd.concat(de_rows)

                s2id = {}
                for i,r in de_rows_minus.iterrows():
                    s = r['Utterance']
                    if not isinstance(s, str):
                        s = s.values[0]
                    de_sentences.append(s)
                    s2id[s] = i
                if len(de_sentences) >0:
                    de_embeddings = laser.embed_sentences(de_sentences, lang='de')
                    en_embedding = laser.embed_sentences([en_utterance], lang='en')
                    most_similar, score = get_most_similar(en_embedding, de_embeddings)
                    ## TODO: threshold for similarity score? 
                    if score > 0.5:
                    #print("de sim row: ")
                        de_row = de_rows.iloc[most_similar]

                if len(de_row) ==0: 
                    logfile.write("no de row found for season {}, episode {} with start time {}\n".format(season, episode, start_en))
                    logfile.write("English: \n")
                    logfile.write(row.to_string() + "\n\n")
                    #print("no de row found for season {}, episode {} with start time {}".format(season, episode, start_en))
                    #print("English: ")
                    #print(row.to_string() + "\n")
                    continue

            de_utterance = de_row['Utterance']
            de_start = de_row.StartTime
            de_end = de_row.EndTime
            if not isinstance(de_utterance, str):
                de_utterance = de_utterance.values[0]
            if not isinstance(de_start, str):
                de_start = de_start.values[0]
            if not isinstance(de_end, str):
                de_end = de_end.values[0]
            print("{},\"{}\",{},{},{},{},{},{},{},{},{},{},{},{}".format(row["Sr No."], de_utterance, en_utterance, de_sentences, speaker, emotion, sentiment, row.Dialogue_ID, row.Utterance_ID, season, episode, de_start, de_end, score))
            #print("{},\"{}\",{},{},{},{},{},{},{},{},{},{}".format(row["Sr No."], de_utterance, speaker, emotion, sentiment, row.Dialogue_ID, row.Utterance_ID, season, episode, de_row.StartTime.values[0], de_row.EndTime.values[0], score))

