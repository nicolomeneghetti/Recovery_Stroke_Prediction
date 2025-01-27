% Nicolò Meneghetti

clear all force; clc; close all force;

warning off;


folder = '.\folder_to_LFPs\'; cd(folder);



fs = 200; % sampling frequency

for rec=1:length(recording_2_be_used) % this is the struct containing the ispi- and contra-lesional LFPs
    
    ipsi = recording_2_be_used(rec).ipsi; % ipsi-lesional LFPs
    contra = recording_2_be_used(rec).contra; % contra-lesional LFPs
    
    % the features are extracted from 30 seconds windows (but this can be
    % changes without any problem)
    window = 30*fs;
    
    % the overal across neighbouring windows. Select the overlap percentage
    overlap_percentage = 0;
    
    overlapLength = floor(window * (overlap_percentage / 100)); % Lunghezza dell'overlap in campioni
    stepSize = window - overlapLength; % Passo tra le finestre consecutive
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % max length duration of LFPs: let's cap the recording to 600 seconds
    secondi_600 = 600*fs;
    
    if(length(ipsi)>secondi_600)
        lenght_extra = (length(ipsi)-secondi_600)/2;
        
        signal_ipsi = ipsi(lenght_extra:(lenght_extra+secondi_600));
        signal_contra = contra(lenght_extra:(lenght_extra+secondi_600));
    else
        signal_ipsi = ipsi;
        signal_contra = contra;
    end
    
    %% Feature selection
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    delta_ipsi = [];  delta_contra = [];
    theta_ipsi = []; theta_contra = [];
    alpha_ipsi = []; alpha_contra = [];
    beta_ipsi = []; beta_contra = [];
    gamma_ipsi = []; gamma_contra = [];
    
    aperiodic_exponent_foof_ipsi = []; aperiodic_exponent_foof_contra = [];
    
    cross_correlation = [];
    
    mutual_information = [];
    
    granger_contra_su_ipsi = []; granger_ipsi_su_contra = [];
    
    
    
    startIndex = 1;
    while (startIndex + window - 1 <= length(signal_ipsi))
        beginning_window = startIndex;
        end_window = startIndex+window-1;
        
        startIndex = startIndex + stepSize;
        
        fprintf('%d of %d - %d of %d \n', rec, length(recording_2_be_used), startIndex, window);
        
        
        % compute the power spectral density
        [pxx_ipsi,f] = pwelch(signal_ipsi(beginning_window:end_window),[],[],[],fs);
        [pxx_contra,f] = pwelch(signal_contra(beginning_window:end_window),[],[],[],fs);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute the powers across bands
        delta_ipsi(end+1)=sum(pxx_ipsi(((f>0.5).*(f<4))>0))/sum(pxx_ipsi);
        theta_ipsi(end+1)=sum(pxx_ipsi(((f>4).*(f<8))>0))/sum(pxx_ipsi);
        alpha_ipsi(end+1)=sum(pxx_ipsi(((f>8).*(f<12))>0))/sum(pxx_ipsi);
        beta_ipsi(end+1)=sum(pxx_ipsi(((f>12).*(f<30))>0))/sum(pxx_ipsi);
        gamma_ipsi(end+1)=sum(pxx_ipsi(((f>30).*(f<50))>0))/sum(pxx_ipsi);
        
        delta_contra(end+1)=sum(pxx_contra(((f>0.5).*(f<4))>0))/sum(pxx_contra);
        theta_contra(end+1)=sum(pxx_contra(((f>4).*(f<8))>0))/sum(pxx_contra);
        alpha_contra(end+1)=sum(pxx_contra(((f>8).*(f<12))>0))/sum(pxx_contra);
        beta_contra(end+1)=sum(pxx_contra(((f>12).*(f<30))>0))/sum(pxx_contra);
        gamma_contra(end+1)=sum(pxx_contra(((f>30).*(f<50))>0))/sum(pxx_contra);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % FOOOF: fitting over one over f trend of PSD
        fooof_results = fooof(f', pxx_ipsi', [0, 100], struct(), true);
        aperiodic_exponent_foof_ipsi(end+1)=fooof_results.background_params(2);
        fooof_results = fooof(f', pxx_contra', [0, 100], struct(), true);
        aperiodic_exponent_foof_contra(end+1)=fooof_results.background_params(2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Cross correlation
        cross_correlation(end+1) = corr(signal_ipsi(beginning_window:end_window), signal_contra(beginning_window:end_window));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % mutual information
        n_equipop_bins = 10;
        ipsi_binned=binr(signal_ipsi(beginning_window:end_window)',finestra,n_equipop_bins,'eqpop')';
        contra_binned=binr(signal_contra(beginning_window:end_window)',finestra,n_equipop_bins,'eqpop')';
        
        vector_response=[]; vector_stimuli=[];
        for c=0:n_equipop_bins-1
            index=find(ipsi_binned==c);
            vector_response=[vector_response;contra_binned(index)];
            vector_stimuli=[vector_stimuli;ipsi_binned(index)];
        end
        [R, nt] = buildr(vector_stimuli, vector_response);
        
        opts.nt=nt; %number of trials
        opts.method='dr'; %direct method
        opts.bias='pt'; % panzeri-trevers correction
        
        [I] = information(R, opts, 'I');
        mutual_information(end+1)=I;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Granger causality metric
        tbl = table(signal_ipsi(beginning_window:end_window),signal_contra(beginning_window:end_window));
        T = size(tbl,1); % Total sample size
        
        numseries = 2;
        numlags = (1:10)';
        nummdls = numel(numlags);
        
        maxp = max(numlags); % Maximum number of required presample responses
        idxpre = 1:maxp;
        idxest = (maxp + 1):T;
        
        EstMdl(nummdls) = varm(numseries,0);
        aic = zeros(nummdls,1);
        
        Y0 = tbl{idxpre,:}; % Presample
        Y = tbl{idxest,:};  % Estimation sample
        for j = 1:numel(numlags)
            Mdl = varm(numseries,numlags(j));
            Mdl.SeriesNames = tbl.Properties.VariableNames;
            EstMdl(j) = estimate(Mdl,Y,'Y0',Y0);
            results = summarize(EstMdl(j));
            aic(j) = results.AIC;
        end
        [~,bestidx] = min(aic);
        p = numlags(bestidx);
        
        BestMdl = EstMdl(bestidx);
        
        [h, summary] = gctest(BestMdl,'Display',false);
        
        granger_contra_su_ipsi(end+1) = table2array(summary(1,4));
        granger_ipsi_su_contra(end+1) = table2array(summary(2,4));


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Cross-frequency coupling
        [cfc_ipsi_theta_gamma(end+1), cfc_contra_theta_gamma(end+1), ....
         cfc_ipsi_delta_gamma(end+1), cfc_contra_delta_gamma(end+1)] = cross_freq_coupling_function(signal_ipsi(beginning_window:end_window),signal_contra(beginning_window:end_window), fs);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % approximateEntropy
        apEntropy_ipsi(end+1) = approximateEntropy(signal_ipsi(beginning_window:end_window));
        apEntropy_contra(end+1) = approximateEntropy(signal_contra(beginning_window:end_window));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Kolmogorov
        kolmogorov_ipsi(end+1)=kolmogorov(signal_ipsi(beginning_window:end_window));
        kolmogorov_contra(end+1)=kolmogorov(signal_contra(beginning_window:end_window));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Hurst Exponent
        hurst_exponent_ipsi(end+1) = estimate_hurst_exponent(signal_ipsi(beginning_window:end_window)');
        hurst_exponent_contra(end+1) = estimate_hurst_exponent(signal_contra(beginning_window:end_window)');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Embedding dimension
        [XR,eLag_ispi,eDim_ispi] = phaseSpaceReconstruction(signal_ipsi(beginning_window:end_window));
        [XR,eLag_contra,eDim_contra] = phaseSpaceReconstruction(signal_contra(beginning_window:end_window));
        
        embeddingDim_ispi(end+1)=eDim_ispi; embeddingDim_contra(end+1)=eDim_contra;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sample entropy
        rangeSampleEntropy_ipsi(end+1)=RangeEn_B(signal_ipsi(beginning_window:end_window), eDim_ispi, 0.2);
        rangeSampleEntropy_contra(end+1)=RangeEn_B(signal_contra(beginning_window:end_window), eDim_contra, 0.2);

    end
    
    
    recording_2_be_used(rec).delta_ipsi = delta_ipsi;
    recording_2_be_used(rec).theta_ipsi = theta_ipsi;
    recording_2_be_used(rec).alpha_ipsi = alpha_ipsi;
    recording_2_be_used(rec).beta_ipsi = beta_ipsi;
    recording_2_be_used(rec).gamma_ipsi = gamma_ipsi;
    
    recording_2_be_used(rec).delta_contra = delta_contra;
    recording_2_be_used(rec).theta_contra = theta_contra;
    recording_2_be_used(rec).alpha_contra = alpha_contra;
    recording_2_be_used(rec).beta_contra = beta_contra;
    recording_2_be_used(rec).gamma_contra = gamma_contra;
    
    recording_2_be_used(rec).cross_correlation = cross_correlation;
    
    recording_2_be_used(rec).granger_contra_su_ipsi = granger_contra_su_ipsi;
    recording_2_be_used(rec).granger_ipsi_su_contra = granger_ipsi_su_contra;
    
    recording_2_be_used(rec).aperiodic_exponent_foof_ipsi = aperiodic_exponent_foof_ipsi;
    recording_2_be_used(rec).aperiodic_exponent_foof_conta = aperiodic_exponent_foof_conta;
    recording_2_be_used(rec).mutual_information = mutual_information;
    
end


%%
folder = '.\feature extraction\'; cd(folder);
save(sprintf('%s\%s', folder, 'features_NO_overlap.mat'), "recording_2_be_used");

