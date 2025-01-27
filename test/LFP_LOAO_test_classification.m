%Nicolò Meneghetti
%% Test best model identified during the validation procedure
clear all; clc; close all; 

% Load and prepare the data produced during the validation procedure
DIR_main = '.\';
filename = 'best_validation_results_across_left_out_animals.mat';

load([DIR_main, filename]);



testing_results = struct([]); 

for acorss_sub_left_out = 1:length(best_validation_across_left_out_animals)
    % for each animal select the best hyperparameters selected from the
    % validation performed on the other animals
    
    left_out_animal = best_validation_across_left_out_animals(acorss_sub_left_out).left_out_animal; 
    classifier = best_validation_across_left_out_animals(acorss_sub_left_out).classifier; 
    window = best_validation_across_left_out_animals(acorss_sub_left_out).window; 
    overlap = best_validation_across_left_out_animals(acorss_sub_left_out).overlap; 
    params = best_validation_across_left_out_animals(acorss_sub_left_out).params;

    % load data
    DIR_main_data = ['.\data\'];
    filename = ['features_',window,'_',overlap,'_2gg.mat'];
    load([DIR_main_data, filename]);
    
    
    % Subject ID
    S = T.animal;
    for i=1:length(S)
        S(i) = cellstr(char(S(i)));
    end
    
    
    % Feature array
    X = table2array(T(:,2:end-1));
    % Label array
    Y = T.label_recover;

    % Split dataset
    %%%%%%%% IDs for test-set = the animal left-out during the validation
    %%%%%%%% is the one we'll be using for the test
    idTest = [cellstr(left_out_animal)];


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

    % select the best model
    nF = 1:size(X,2);
    hyperparameters = []; 
    switch classifier
        case 'ANN'
            mdl = 'ann';
            hyperparameters.layerSizes = {params};
        case 'SVMlin'
            mdl = 'svm';
            hyperparameters.kernel = 'linear';

            % this folder contains the results of the validation for all
            % the tested set of hyparameters
            dir_lod_folder = '.\validation_results_leave_one_out\'; 
            filename_lod_folder = ['Results_',left_out_animal,'_',classifier,'_',window,'_',overlap,'.mat']; 
            load([dir_lod_folder, filename_lod_folder]); 

            hyperparameters.order = [];
            hyperparameters.C = Results_tot{best_validation_across_left_out_animals(acorss_sub_left_out).idx,2}; 
            hyperparameters.gamma = Results_tot{best_validation_across_left_out_animals(acorss_sub_left_out).idx,3}; 

        case 'SVMrbf'
            mdl = 'svm';
            hyperparameters.kernel = 'rbf';
            
            % this folder contains the results of the validation for all
            % the tested set of hyparameters
            dir_lod_folder = '.\validation_results_leave_one_out\';
            filename_lod_folder = ['Results_',left_out_animal,'_',classifier,'_',window,'_',overlap,'.mat'];
            load([dir_lod_folder, filename_lod_folder]);

            hyperparameters.order = [];
            hyperparameters.C = Results_tot{best_validation_across_left_out_animals(acorss_sub_left_out).idx,2};
            hyperparameters.gamma = Results_tot{best_validation_across_left_out_animals(acorss_sub_left_out).idx,3};
    end

    X_train = X(:,nF);
    Y_train = Y;
    S_train = S;
    X_test = X_leftout(:,nF);
    Y_test = Y_leftout;
    S_test = S_leftout;
    H = T.Properties.VariableNames(2:end);

    % we compute the test accuracy across 100 iterations
    nIter = 100;
    PREDS = zeros([nIter, length(unique(S_test))]);
    confusion_matrix_subjects = []; 
    confusion_matrix_windows = []; 
    for i = 1:nIter 
        disp(['Processing ',num2str(acorss_sub_left_out),' of ',num2str(length(best_validation_across_left_out_animals)),': Test ', num2str(i), ' of ', num2str(nIter)]);
        [Accuracy, Accuracy_subj, ConfMat, ConfMat_subj, Y_epochs, Y_subj, id] = test_classification_leftout(X_train, Y_train, X_test, Y_test, H, S_test, mdl, hyperparameters);
        acc(i,:) = [Accuracy, Accuracy_subj];
        confusion_matrix_windows = cat(3,confusion_matrix_windows, ConfMat); 
        confusion_matrix_subjects= cat(3,confusion_matrix_subjects, ConfMat_subj); 
    end
    
    % let's append the results of the classification on the test set
    testing_results(end+1).left_out_animal=left_out_animal; 
    testing_results(end).Accuracy=acc; 
    testing_results(end).Accuracy_window_mean=median(acc(:,1)); 
    testing_results(end).Accuracy_subject_mean=median(acc(:,2)); 
    testing_results(end).label_of_left_out_animal=mean(Y_test); 
    testing_results(end).confusion_matrix_subjects=sum(confusion_matrix_subjects,3); 
    testing_results(end).confusion_matrix_windows=sum(confusion_matrix_windows,3); 

end

% save the results
save('.\testing_results_2gg.mat', 'testing_results');


% compute the confusion matrix
cm_sub = zeros(2,2);
cm_win = zeros(2,2);
for i=1:length(testing_results)
    cm_sub = cm_sub +testing_results(i).confusion_matrix_subjects/100;
    cm_win = cm_win+testing_results(i).confusion_matrix_windows/100;
end

print('this is the confusion matrix : \n')
disp(cm_sub)


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
