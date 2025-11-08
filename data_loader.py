import os, json

def gather_datas():
    content = ""
    
    for name in os.listdir("datas"):
        if os.path.isfile(os.path.join("datas", name)):
            with open(os.path.join("datas", name), "r", encoding="utf-8") as f:

                data = json.load(f)
                messages_count = len(data.keys())

                for i in range(messages_count):
                    index = messages_count - i - 1
                    key = "msg" + str(index)
                    content += "<|who_i_am|>" + data[key]["author"] + "<|end_who_i_am|>" + "<|bos|>" + data[key]["content"] + "<|eos|>" + "\n"

    return content

def gather_messy_datas():
    """
    this is probably a better way to get the datas anyway
    """
    content = ""
    
    for name in os.listdir("datas"):
        if os.path.isfile(os.path.join("datas", name)):
            with open(os.path.join("datas", name), "r", encoding="utf-8") as f:

                data = json.load(f)
                
                for key in data.keys():
                    content += "<|who_i_am|>" + data[key]["author"] + "<|end_who_i_am|>" + "<|bos|>" + data[key]["content"] + "<|eos|>" + "\n"

    return content