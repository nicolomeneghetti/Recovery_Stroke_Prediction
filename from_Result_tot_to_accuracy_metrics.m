function [Summary_NICO] = from_Result_tot_to_accuracy_metrics(Results_tot, metrics)

Perf_win = nan(size(Results_tot,1),1);
Perf_subj = nan(size(Results_tot,1),1);

if strcmp(metrics, 'accuracy') || strcmp(metrics, 'corr')
    
    for j = 1:size(Results_tot)
        if isa(Results_tot{j,end-1}, 'double')
            Perf_win(j,1) = Results_tot{j,end-1};
        else
            Perf_win(j,1) = cell2mat(Results_tot{j,end-1});
        end
        
        if isa(Results_tot{j,end}, 'double')
            Perf_subj(j,1) = Results_tot{j,end};
        else
            Perf_subj(j,1) = cell2mat(Results_tot{j,end});
        end
    end
    
    [m, idx] = max(Perf_subj);
    %check if multiple max exist
    if(sum(Perf_subj==m)>1)
        index_maximal_subject = Perf_subj == m;
        Perf_win(index_maximal_subject==0) = 0;
        [m, idx] = max(Perf_win);
        m = Perf_subj(idx);
        if(sum(Perf_win==m)>1)
            indici_massima_finestra = find(Perf_win==m);
            idx = indici_massima_finestra(randperm(sum(Perf_win==m),1));
        end
        
    end
    
    
    features = Results_tot{idx,1};
    params = Results_tot{idx,2};
    
    Summary= [num2cell(idx), num2cell(m)];
    Summary_NICO = [idx, m, Results_tot{idx,2}, Results_tot{idx,end-1}];
elseif strcmp(metrics, 'rmse')
    
    for j = 1:size(Results_tot)
        if isa(Results_tot{j,end-3}, 'double')
            Perf_win(j,1) = Results_tot{j,end-3};
        else
            Perf_win(j,1) = cell2mat(Results_tot{j,end-3});
        end
        
        if isa(Results_tot{j,end-2}, 'double')
            Perf_subj(j,1) = Results_tot{j,end-2};
        else
            Perf_subj(j,1) = cell2mat(Results_tot{j,end-2});
        end
    end
    
    [m, idx] = min(Perf_subj);
    %check if multiple min exist
    if(sum(Perf_subj==m)>1)
        index_minimal_subject = Perf_subj == m;
        Perf_win(index_minimal_subject==0) = inf;
        [m, idx] = min(Perf_win);
        m = Perf_subj(idx);
        if(sum(Perf_win==m)>1)
            indici_massima_finestra = find(Perf_win==m);
            idx = indici_massima_finestra(randperm(sum(Perf_win==m),1));
        end
    end
    
    features = Results_tot{idx,1};
    params = Results_tot{idx,2};
    
    disp(['Best LOSO-CV results for ', filename, ' : ' ...
        'RMSE = ', num2str(m)]);
    Summary = [cellstr(filename), num2cell(idx), num2cell(m)];
    Summary_NICO = [idx, m, Results_tot{idx,2}, Results_tot{idx,end-1}];
    
    
end
end