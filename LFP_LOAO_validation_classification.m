% Nicolo Meneghetti

clear all force;
close all force;
clc;

home_directory = cd; 
%%
% Set of tested classifiers
classifiers = [cellstr('SVMrbf'); cellstr('SVMlin'); cellstr('SVMquad'); ...
    cellstr('SVMcub'); cellstr('RF'); cellstr('kNN');];

% Set of tested windows length
windows = [cellstr('30_seconds'), cellstr('60_seconds')];

% Set of tested overlap between neighbouring windows
overlaps = [cellstr('50_overlap'), cellstr('NO_overlap'), cellstr('random_windows')];

% This code is performed for the classification of motor recovery class from post-stroke LFPs. To change it to 
% baseline recording change this parameter to 'baseline'
time_relative_to_stroke = '2gg';

% vector of animal names used in this study (the names are arbitrary)
animal_vector = cell({}); 
animal_vector{1}='347_0';
animal_vector{2}='347_sx';
animal_vector{3}='349';
animal_vector{4}='375_0';
animal_vector{5}='398_0';
animal_vector{6}='398_sx';
animal_vector{7}='438';
animal_vector{8}='454_0';
animal_vector{9}='454_dx';
animal_vector{10}='454_sx';
animal_vector{11}='475_sx';
animal_vector{12}='477A_0';
animal_vector{13}='514_sx';
animal_vector{14}='534_0';
animal_vector{15}='534_sx';
animal_vector{16}='cbs130_dx';


final_list_results = []; 
%% main for loop: the LOAO loops across the animal: one of them is treated as the test animal in each iteration
for animale_leave_1_out = 1:length(animal_vector)
    disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%']);
    disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%']);
    disp(['LEAVING OUT ANIMAL ' num2str(animale_leave_1_out) ' of ' num2str(length(animal_vector))]);
    disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%']);
    disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%']);
    %loop across overlaps
    for iii = 1:length(overlaps)
        %loop across window lengths
        for jjj = 1:length(windows)
            %loop across classifiers
            for kkk = 1:length(classifiers)
    
                overlap = overlaps{iii};
                window = windows{jjj};
                classifier = classifiers{kkk};
                
                %% Load and prepare data
                DIR_main = ['folders_to_data\'];
                filename = ['features_', window,'_', overlap, '_',time_relative_to_stroke , '.mat'];

                
                load([DIR_main, filename]);
                
                % Subject ID
                S = T.animal;
                for i=1:length(S)
                    S(i) = cellstr(char(S(i)));
                end
                
                % Feature array
                X = table2array(T(:,2:end-1));
                % Label array
                Y = T.label_recover;
                
                %% Split dataset
                % leave one animal out : LOAO
                idTest = [cellstr(animal_vector{animale_leave_1_out})]; 
                
                S_leftout = cell(0);
                X_leftout = [];
                Y_leftout = [];
                
                for i = 1:length(idTest)
                    isubj = strcmp(S, idTest(i));
                    S_leftout = [S_leftout; S(isubj)];
                    S(isubj) = [];
                    X_leftout = [X_leftout; X(isubj,:)];
                    X(isubj,:) = [];
                    Y_leftout = [Y_leftout; Y(isubj)];
                    Y(isubj) = [];
                end
                
                
                selectedFeatures = {1:size(X,2)}; % we select ALL the features in the dataset
                
                %% Feature selection and classification (Training/Validation LOSO-CV)
                Results_tot = cell(0);
                for i = 1:length(selectedFeatures)
                
                    nF = selectedFeatures{i}; 
                
                    X_trainval = X(:,nF);
                    Y_trainval = Y;
                    S_trainval = S;
                    H = T.Properties.VariableNames(2:end);
                
                    disp(['Feature combination n. ' num2str(i) ' of ' num2str(length(selectedFeatures)) '...']);
    
                    if strcmp(classifier, 'SVMrbf')
                        Results = classify_SVM(X_trainval, Y_trainval, H, S_trainval, 'rbf', []); % Support Vector Regression  
                    elseif strcmp(classifier, 'SVMlin')
                        Results = classify_SVM(X_trainval, Y_trainval, H, S_trainval, 'linear', []); % Support Vector Regression 
                    elseif strcmp(classifier, 'SVMquad')
                        Results = classify_SVM(X_trainval, Y_trainval, H, S_trainval, 'polynomial', 2); % Support Vector Regression 
                    elseif strcmp(classifier, 'SVMcub')
                        Results = classify_SVM(X_trainval, Y_trainval, H, S_trainval, 'polynomial', 3); % Support Vector Regression 
                    elseif strcmp(classifier, 'RF')
                        Results = classify_RF(X_trainval, Y_trainval, H, S_trainval); % Random Forest    
                    elseif strcmp(classifier, 'ANN')
                        Results = classify_ANN(X_trainval, Y_trainval, H, S_trainval); % Neural Networks
                    elseif strcmp(classifier, 'logreg')
                        Results = classify_logreg(X_trainval, Y_trainval, H, S_trainval); % Logistic Regression  
                    elseif strcmp(classifier, 'kNN')
                        Results = classify_kNN(X_trainval, Y_trainval, H, S_trainval); % k Nearest Neighbors
                    end
    
                    Results = [repmat({nF},[size(Results,1),1]), num2cell(Results)];
                    Results_tot = [Results_tot; Results];
                
                end
    
                
                final_list_results(end+1).left_out_animal = animal_vector{animale_leave_1_out}; 
                final_list_results(end).classifier = classifier; 
                final_list_results(end).window = window; 
                final_list_results(end).overlap = overlap; 
                
                [Summary_accuracy_metrics] = from_Result_tot_to_accuracy_metrics(Results_tot, 'accuracy'); %custom-made function to compute the accuracy metrics
                if(iscell(Summary_accuracy_metrics(1)))
                    final_list_results(end).idx = cell2mat(Summary_accuracy_metrics(1));
                    final_list_results(end).accuracy_subject = cell2mat(Summary_accuracy_metrics(2));
                    final_list_results(end).params = cell2mat(Summary_accuracy_metrics(3));
                    final_list_results(end).accuracy_window = cell2mat(Summary_accuracy_metrics(4));
                else
                    final_list_results(end).idx = Summary_accuracy_metrics(1);
                    final_list_results(end).accuracy_subject = Summary_accuracy_metrics(2);
                    final_list_results(end).params = Summary_accuracy_metrics(3);
                    final_list_results(end).accuracy_window = Summary_accuracy_metrics(4);
                end
    
            end
    
        end
    
    end
    
end


save([home_directory,'\', 'validation_results_leave_one_animal_out.mat'], 'final_list_results');

best_validation_across_left_out_animals = struct([]); 
for an=1:length(animal_vector) %for each animal, select the set of hyperparameters maximizing the validation accuracy
    indexes_animal = arrayfun(@(x) strcmp(x.left_out_animal,animal_vector{an}),final_list_results); 
    rows_animal = final_list_results(indexes_animal==1); 

    [~, idx] = sort(-[rows_animal.accuracy_subject]);
    sortedStruct = rows_animal(idx);

    %if there are multiple maximal accuracy on subject select the one with
    %mazimal window accuracy
    indexes_max_sub = arrayfun(@(x) (x.accuracy_subject == sortedStruct(1).accuracy_subject),sortedStruct); 
    if(sum(indexes_max_sub)>1)
        max_rows = sortedStruct(indexes_max_sub==1); 
        [nothing,index_max_window] = max([max_rows.accuracy_window]); 
        best_validation_across_left_out_animals(end+1).left_out_animal=max_rows(index_max_window).left_out_animal; 
        best_validation_across_left_out_animals(end).classifier=max_rows(index_max_window).classifier; 
        best_validation_across_left_out_animals(end).window=max_rows(index_max_window).window; 
        best_validation_across_left_out_animals(end).overlap=max_rows(index_max_window).overlap; 
        best_validation_across_left_out_animals(end).idx=max_rows(index_max_window).idx; 
        best_validation_across_left_out_animals(end).accuracy_subject=max_rows(index_max_window).accuracy_subject; 
        best_validation_across_left_out_animals(end).params=max_rows(index_max_window).params; 
        best_validation_across_left_out_animals(end).accuracy_window=max_rows(index_max_window).accuracy_window; 
    else
        best_validation_across_left_out_animals(end+1).left_out_animal=sortedStruct(indexes_max_sub==1).left_out_animal; 
        best_validation_across_left_out_animals(end).classifier=sortedStruct(indexes_max_sub==1).classifier; 
        best_validation_across_left_out_animals(end).window=sortedStruct(indexes_max_sub==1).window; 
        best_validation_across_left_out_animals(end).overlap=sortedStruct(indexes_max_sub==1).overlap; 
        best_validation_across_left_out_animals(end).idx=sortedStruct(indexes_max_sub==1).idx; 
        best_validation_across_left_out_animals(end).accuracy_subject=sortedStruct(indexes_max_sub==1).accuracy_subject; 
        best_validation_across_left_out_animals(end).params=sortedStruct(indexes_max_sub==1).params; 
        best_validation_across_left_out_animals(end).accuracy_window=sortedStruct(indexes_max_sub==1).accuracy_window;         
    end
end

% save the set of best hyperparameters for each of the tested animal
save([home_directory,'\', 'best_validation_results_across_left_out_animals.mat'], 'best_validation_across_left_out_animals');



%% FUNCTION - Optimize parameters for SVM

function RESULTS = classify_SVM(X, Y, H, S, kernel, order)

    % Set specific hyperparameters
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    
    % Specify the kernel function
    hyperparameters.kernel = kernel;
    hyperparameters.order = order;

    i = 1;
    
    for ii = 1:length(C)
        for jj = 1:length(gamma)
            disp([num2str(i) ' of ' num2str(length(C)*length(gamma)) '...']);
            Hyper(i,:) = [C(ii), gamma(jj)];
            hyperparameters.C = C(ii);
            hyperparameters.gamma = gamma(jj);        
            [AccuracyLOSO(i,1), CV_errorLOSO(i,1), ConfMatLOSO(:,:,i), ConfMatSubj(:,:,i)] = test_classifier_LOSO(X, Y, H, S, 'svm', hyperparameters);
            Acc_subj(i,1) = sum(diag(ConfMatSubj(:,:,i)))/sum(sum(ConfMatSubj(:,:,i))) * 100; 
            disp(['Accuracy (LOSO): ', num2str(AccuracyLOSO(i,1)), '% - Accuracy (Subject): ', num2str(Acc_subj(i,1)), '%']);
            i = i + 1;
        end
    end
    
    % Results
    RESULTS = [Hyper, CV_errorLOSO, AccuracyLOSO, Acc_subj];

end

%% FUNCTION - Optimize parameters for Logistic Regression

function RESULTS = classify_logreg(X, Y, H, S)

    % Set specific hyperparameters
    th = 0.2:0.05:0.8;
   
    i = 1;
    
    for ii = 1:length(th)
        disp([num2str(i) ' of ' num2str(length(th)) '...']);
        Hyper(i,:) = th(ii);
        hyperparameters.th = th(ii);
        [AccuracyLOSO(i,1), CV_errorLOSO(i,1), ConfMatLOSO(:,:,i), ConfMatSubj(:,:,i)] = test_classifier_LOSO(X, Y, H, S, 'logreg', hyperparameters);
        Acc_subj(i,1) = sum(diag(ConfMatSubj(:,:,i)))/sum(sum(ConfMatSubj(:,:,i))) * 100;     
        disp(['Accuracy (LOSO): ', num2str(AccuracyLOSO(i,1)), '% - Accuracy (Subject): ', num2str(Acc_subj(i,1)), '%']); 
        i = i + 1;
    end
    
    % Results
    RESULTS = [Hyper, CV_errorLOSO, AccuracyLOSO, Acc_subj];

end

%% FUNCTION - Optimize parameters for k-NN classifier

function RESULTS = classify_kNN(X, Y, H, S)

    % Set specific hyperparameters
    k = [3, 5, 7, 9];
    hyperparameters.distance = 'euclidean';
    hyperparameters.distweight = 'equal'; 

    i = 1;
    
    for ii = 1:length(k)    
        disp([num2str(i) ' of ' num2str(length(k)) '...']);
        Hyper(i,:) = k(ii);
        hyperparameters.k = k(ii);        
        [AccuracyLOSO(i,1), CV_errorLOSO(i,1), ConfMatLOSO(:,:,i), ConfMatSubj(:,:,i)] = test_classifier_LOSO(X, Y, H, S, 'knn', hyperparameters);
        Acc_subj(i,1) = sum(diag(ConfMatSubj(:,:,i)))/sum(sum(ConfMatSubj(:,:,i))) * 100;       
        disp(['Accuracy (LOSO): ', num2str(AccuracyLOSO(i,1)), '% - Accuracy (Subject): ', num2str(Acc_subj(i,1)), '%']);
        i = i + 1;
    end
    
    % Results
    RESULTS = [Hyper, CV_errorLOSO, AccuracyLOSO, Acc_subj];

end

%% FUNCTION - Optimize parameters for Random Forest

function RESULTS = classify_RF(X, Y, H, S)

    % Set specific hyperparameters
    trees = [10,20,50,100,200,500,1000];

    i = 1;
    
    for ii = 1:length(trees)    
        disp([num2str(i) ' of ' num2str(length(trees)) '...']);
        Hyper(i,:) = trees(ii);
        hyperparameters.trees = trees(ii);        
        [AccuracyLOSO(i,1), CV_errorLOSO(i,1), ConfMatLOSO(:,:,i), ConfMatSubj(:,:,i)] = test_classifier_LOSO(X, Y, H, S, 'rf', hyperparameters);
        Acc_subj(i,1) = sum(diag(ConfMatSubj(:,:,i)))/sum(sum(ConfMatSubj(:,:,i))) * 100;  
        disp(['Accuracy (LOSO): ', num2str(AccuracyLOSO(i,1)), '% - Accuracy (Subject): ', num2str(Acc_subj(i,1)), '%']);
        i = i + 1;
    end
        
    % Results
    RESULTS = [Hyper, CV_errorLOSO, AccuracyLOSO, Acc_subj];

end

%% FUNCTION - Optimize parameters for Neural Networks

function RESULTS = classify_ANN(X, Y, H, S)

    % Set specific hyperparameters
    base = [5,10,15,20,25]; % units per layer
    layers1 = cell(0);
    layers2 = cell(0);
    layers3 = cell(0);

    % Set 1-layer ANN
    n = 1;
    for i = 1:length(base)
        layers1(n,1) = {base(i)};
        n = n + 1;
    end
    % Set 2-layer ANN
    n = 1;
    for i = 1:length(base)
        for j = 1:length(base)
            layers2(n,1) = {[base(i), base(j)]};
            n = n + 1;
        end
    end
    % Set 3-layer ANN
    n = 1;
    for i = 1:length(base)
        for j = 1:length(base) 
            for k = 1:length(base)
                layers3(n,1) = {[base(i), base(j), base(k)]};
                n = n + 1;
            end
        end
    end

    layerSizes = [layers1; layers2; layers3];
    i = 1;
    
    for ii = 1:length(layerSizes)

        disp([num2str(i) ' of ' num2str(length(layerSizes)) '...']);
        Hyper(i,:) = layerSizes(ii);
        hyperparameters.layerSizes = layerSizes(ii);           
        [AccuracyLOSO(i,1), CV_errorLOSO(i,1), ConfMatLOSO(:,:,i), ConfMatSubj(:,:,i)] = test_classifier_LOSO(X, Y, H, S, 'ann', hyperparameters);
        Acc_subj(i,1) = sum(diag(ConfMatSubj(:,:,i)))/sum(sum(ConfMatSubj(:,:,i))) * 100;   
        disp(['Accuracy (LOSO): ', num2str(AccuracyLOSO(i,1)), '% - Accuracy (Subject): ', num2str(Acc_subj(i,1)), '%']);
        i = i + 1;

    end   
    
    % Results
    RESULTS = [Hyper, num2cell(CV_errorLOSO), num2cell(AccuracyLOSO), num2cell(Acc_subj)];

end
