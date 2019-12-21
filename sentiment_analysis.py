#encoding:utf-8
from encoder import Model, sst_binary, model_train, model_predict


max_length = 100
load_path = 'data/chinese'
language = 'chinese'
tr_num = 17000
va_num = 2000


model = Model(max_length)

all_data = sst_binary(load_path)  #分别获取所有的句子和标签
print('=> Succeeds in loading <' + language + '> file and starting to translate words into Embeddedness······')

x, y, wi = model.transform(all_data)  #将每个句子里的词转化成词频索引值
print('=> Succeeds in translating swords into word Embeddedness and starting to train the model process······')

accuracy = model_train(x, y, wi, language, max_length, tr_num, va_num)  #训练模型
print('=> accuracy: ', accuracy*100, '%')

while True:
    sentence = input("Please enter a single sentence to predict:")
    result = model_predict(sentence, max_length)
    if result == 0:
       print('negative')
    else:
       print("positive")
