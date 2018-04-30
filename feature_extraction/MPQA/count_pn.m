clear all;

pn_processing;

% fileID = fopen('new_omg_TrainTranscripts1.csv');
% fileID = fopen('new_omg_ValidationTranscripts1.csv');
fileID = fopen('new_omg_TestTranscripts_v3.csv');
data_all_train = textscan(fileID,'%s %s %s', 'Delimiter',',');
data_all = data_all_train;
fclose(fileID);


% [fid,errmsg] = fopen('new_omg_TrainTranscripts_features.txt', 'w');
% [fid,errmsg] = fopen('new_omg_ValidationTranscripts_features.txt', 'w');
[fid,errmsg] = fopen('new_omg_TestTranscripts_features.txt', 'w');

num_u = length(data_all{1});
num_pos = [];
num_neg = [];
num_words_u = [];
for ui = 1: num_u
    txt_this = data_all{3}{ui};
    if isempty(txt_this)
        num_words_u = [num_words_u; 0];
    else
        C = strsplit(txt_this);
        num_words_u = [num_words_u; length(C)];
    end
    C = strsplit(txt_this);
    num_pos_1u = 0;
    num_neg_1u = 0;
    for word_i = 1: length(C)
        IndexC = strcmp(words_all, C{word_i});
        if sum(IndexC)>0
            pn_value = unique(pn_all(IndexC));
%             disp(C{word_i});
            for pn_i = 1:length(pn_value)
                pn_value_this = pn_value(pn_i);
                if pn_value_this ==1 
                    num_pos_1u = num_pos_1u+1;
                elseif pn_value_this == -1
                    num_neg_1u = num_neg_1u+1;
                end
            end
        end
    end
    num_pos = [num_pos; num_pos_1u];
    num_neg = [num_neg; num_neg_1u];
end


%% video level processing
num_u = length(data_all{1});
video_init = data_all{1}{2};% the name of the first video
count_video_init = 1;
video_flag = [];
video_flag_this = 1;
for ui = 1: num_u
    video_this = data_all{1}{ui};
    if strcmp(video_init, video_this)
        video_flag = [video_flag; video_flag_this];
    else
        video_init = video_this;
        video_flag_this = video_flag_this+1;
        video_flag = [video_flag; video_flag_this];
    end
end


num_pos_v = []; 
num_neg_v = [];
num_video = max(video_flag);
for vi = 1: num_video
    index = find(video_flag == vi);
    num_u_1video = length(index);
    num_pos_v_this = sum(num_pos(index));
    num_pos_v = [num_pos_v; num_pos_v_this*ones(num_u_1video, 1)];
    num_neg_v_this = sum(num_neg(index));
    num_neg_v = [num_neg_v; num_neg_v_this*ones(num_u_1video, 1)];
end


% write into file
for ui = 1: num_u
    fprintf(fid, '%s ', data_all{1}{ui});
    fprintf(fid, '%s ', data_all{2}{ui});
    fprintf(fid, '%d %d %d %d %d\n', num_pos(ui), num_neg(ui), num_pos_v(ui), num_neg_v(ui), num_words_u(ui));
end

