import os, json, random

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

def convert_datas_to_tokens(tokenizer):
    """
    Convert the datas to tokens
    Could have used #gather_messy_datas but i want the file to be saved every so often
    """

    content = []

    save_counter = 0

    for name in os.listdir("datas"):
        if os.path.isfile(os.path.join("datas", name)):
            with open(os.path.join("datas", name), "r", encoding="utf-8") as f:
                data = json.load(f)

                for key in data.keys():
                    raw = "<|who_i_am|>" + data[key]["author"] + "<|end_who_i_am|>" + "<|bos|>" + data[key]["content"] + "<|eos|>"

                    tokens = tokenizer.encode(raw)
                    content.extend(tokens)

                    if save_counter % 1000 == 0:
                        jdata = {
                            "tokens": content
                        }
                        with open(os.path.join("datas\\tokenized", "tokens.json"), "w", encoding="utf-8") as output:
                            output.write(json.dumps(jdata))


                    save_counter+=1

    jdata = {
        "tokens": content
    }
    with open(os.path.join("datas\\tokenized", "tokens.json"), "w", encoding="utf-8") as output:
        output.write(json.dumps(jdata))

def convert_datas_to_tokens_with_shuffle(tokenizer, random_val=10):
    content = []

    save_counter = 0

    for name in os.listdir("datas"):
        if os.path.isfile(os.path.join("datas", name)):
            with open(os.path.join("datas", name), "r", encoding="utf-8") as f:
                data = json.load(f)

                for key in data.keys():
                    raw = "<|who_i_am|>" + data[key]["author"] + "<|end_who_i_am|>" + "<|bos|>" + data[key]["content"] + "<|eos|>"

                    tokens = tokenizer.encode(raw)
                    
                    if random.randint(0,random_val) == 0:
                        temp = list(tokens)
                        temp.extend(content)
                        content = temp
                    else:
                        content.extend(tokens)

                    if save_counter % 1000 == 0:
                        jdata = {
                            "tokens": content
                        }
                        with open(os.path.join("datas\\tokenized", "tokens.json"), "w", encoding="utf-8") as output:
                            output.write(json.dumps(jdata))


                    save_counter+=1

    jdata = {
        "tokens": content
    }
    with open(os.path.join("datas\\tokenized", "tokens.json"), "w", encoding="utf-8") as output:
        output.write(json.dumps(jdata))
    return content

def get_data_tokens_as_list():
    with open(os.path.join("datas\\tokenized", "tokens.json"), "r", encoding="utf-8") as f:
        return json.load(f).get("tokens")
                    

from fastbpe import Tokenizer
if __name__ == '__main__':
    convert_datas_to_tokens(Tokenizer(24576))
    # convert_datas_to_tokens_with_shuffle(Tokenizer(24576), random_val=8)