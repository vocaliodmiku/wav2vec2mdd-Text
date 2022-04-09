def get(lines, wav_id):
    data = {}
    length = len(lines)
    index = 0
    while index < length-4:
        
        

def get2(lines, wav_id):
    data = {}
    length = len(lines)
    index = 0
    while index < length-4:
        _wav_id = lines[index].strip().split(' ')[0]
        if wav_id == _wav_id: 
            #print(lines[index].strip())
            print(lines[index+1].strip())
            print(lines[index+2].strip())
            #print(lines[index+3].strip())
            return 
        else:
            index += 1

ref_human = open("ref_human_detail_LV60", 'r').readlines()
large = open("human_our_detail_LARGE", 'r').readlines()
lv60 = open("human_our_detail_LV60", 'r').readlines()
xlsr = open("human_our_detail_XLSR", 'r').readlines()
