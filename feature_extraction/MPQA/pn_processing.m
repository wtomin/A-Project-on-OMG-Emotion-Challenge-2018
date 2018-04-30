

fileID = fopen('subjclueslen1-HLTEMNLP05_1.txt');
data_all = textscan(fileID,'%s %s %s %s %s %s');
fclose(fileID);
%1:weak/strong
%2:length
%3:word
%4:stemmed
%5:positive/negative/neutral/both

% remove the title of each elment
num_words = length(data_all{3});
words_all_raw = data_all{3};
pn_all_raw = data_all{6};
words_all = [];
pn_all = [];
for i = 1:num_words
    words_all{end+1}= words_all_raw{i}(7:end);
    s = pn_all_raw{i}(17);
    if s == 's'
        pn_all = [pn_all; 1];
    elseif s == 'g'
        pn_all = [pn_all; -1];
    elseif s == 'u'
        pn_all = [pn_all; 0];
    elseif s == 't'
        pn_all = [pn_all; 0.1];
    else
%         disp(pn_all_raw{i});
        pn_this = input([pn_all_raw{i}, '\n']);
        pn_all = [pn_all; pn_this];
    end
end
data_all_dic = data_all;


