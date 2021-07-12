import srt
import os, sys
import argparse
import pandas
import re
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--de-srt', type=str, metavar='PATH', help='path to german subtitles (srt)')
    parser.add_argument('--season', type=int,  help='season of srt')
    parser.add_argument('--episode', type=int,  help='episode of srt')    
    args = parser.parse_args()

    with open(args.de_srt, 'r') as f:
        de_srt_lst = list(srt.parse(f.read()))

    print("Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime")
    for i,sub in enumerate(de_srt_lst):
        #print(sub)
        content = sub.content.replace("\n", " ")
        content = content.replace('"', "'")
        content = '"' + content +'"'
        #start = get_time_in_s(datetime.datetime.strptime(row.StartTime, '%H:%M:%S,%f'))
        line = ','.join([str(sub.index),content,"-","-","-","-","-",str(args.season), str(args.episode),str(sub.start).split(".")[0], str(sub.end).split(".")[0]])
        print(line)

        


