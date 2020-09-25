import math,sys,os,glob

DIR="../data/ud-treebanks-v2.5/"

lgs=os.listdir(DIR)
exceptions=["UD_Armenian-ArmTDP","UD_French-Sequoia","UD_Finnish-FTB","UD_Ukrainian-IU","UD_Dutch-LassySmall","UD_Polish-PDB","UD_Portuguese-Bosque","UD_Italian-PoSTWITA","UD_Old_French-SRCMF","UD_Italian-ISDT","UD_Latvian-LVTB","UD_Lithuanian-ALKSNIS","UD_Galician-CTG","UD_Vietnamese-VTB","UD_Greek-GDT","UD_Spanish-GSD","UD_Catalan-AnCora","UD_English-EWT","UD_Russian-SynTagRus","UD_Czech-PDT","UD_Italian-VIT","UD_Marathi-UFAL","UD_French-FTB","UD_Tamil-TTB","UD_Spanish-AnCora","UD_Finnish-TDT","UD_French-ParTUT","UD_Turkish-IMST","UD_Swedish-Talbanken","UD_Arabic-NYUAD","UD_Italian-ParTUT","UD_Czech-FicTree","UD_Wolof-WTB","UD_Italian-TWITTIRO","UD_Czech-CAC","UD_German-GSD","UD_French-Spoken","UD_Persian-Seraji","UD_Czech-CLTT","UD_French-GSD","UD_Dutch-Alpino","UD_Coptic-Scriptorium","UD_Scottish_Gaelic-ARCOSG","UD_Portuguese-GSD","UD_Old_Russian-TOROT","UD_Arabic-PADT","UD_Hebrew-HTB","UD_English-ParTUT","UD_German-HDT","UD_Japanese-BCCWJ"]
lgs=[l for l in lgs if l not in exceptions]

for language in lgs:#["UD_Basque-BDT","UD_Japanese-GSD","UD_Serbian-SET"]:
    files=glob.glob(DIR+"/"+language+"/*.conllu")
    if len(files)>2:
        print(language)
        for f in files:
            if f.split(".")[-2].split("-")[-1]=="train":
                train_file=f
            elif f.split(".")[-2].split("-")[-1]=="test":
                test_file=f
        train=[l.strip().split() for l in open(train_file).readlines() if l[0]!="#"]
        test=[l.strip().split() for l in open(test_file).readlines() if l[0]!="#"]

        for conll in [test]: # train
            sents=0
            total_length=0
            sent_len=0
            for line in conll:
                if len(line)>1:
                    sent_len+=1
                else:
                    total_length+=sent_len
                    sents+=1
                    sent_len=0

            print("Av sentlength:",total_length/sents)
