function [cfc_theta_cor_striatum, cfc_theta_striatum_cor] = cross_freq_coupling_NICO(yy_cortex,yy_striatum, fs)
    
    bpfilt_THETA = designfilt('bandpassiir', ...
        'FilterOrder',20,'HalfPowerFrequency1',2.5, ...
        'HalfPowerFrequency2',4.5,'SampleRate',fs);
    yy_cortex_THETA = filtfilt(bpfilt_THETA,yy_cortex);
    yy_striatum_THETA = filtfilt(bpfilt_THETA,yy_striatum);

%     bpfilt_DELTA = designfilt('bandpassiir', ...
%         'FilterOrder',20,'HalfPowerFrequency1',0.5, ...
%         'HalfPowerFrequency2',4,'SampleRate',fs);
%     yy_ipsi_DELTA = filtfilt(bpfilt_DELTA,yy_ipsi);
%     yy_contra_DELTA = filtfilt(bpfilt_DELTA,yy_contra);
% 
%     bpfilt_GAMMA = designfilt('bandpassiir', ...
%         'FilterOrder',20,'HalfPowerFrequency1',30, ...
%         'HalfPowerFrequency2',50,'SampleRate',fs);
%     yy_ipsi_GAMMA = filtfilt(bpfilt_GAMMA,yy_ipsi);
%     yy_contra_GAMMA = filtfilt(bpfilt_GAMMA,yy_contra);

    phi = angle(hilbert(yy_cortex_THETA));
    amp = abs(hilbert(yy_striatum_THETA));
    p_bins = linspace(-pi,pi,63);
    a_mean = zeros(size(p_bins)-1);
    p_mean = zeros(size(p_bins)-1);
    for kcfc = 1:length(p_bins)-1
        pL = p_bins(kcfc);
        pR = p_bins(kcfc+1);
        indices=(phi>=pL) & (phi<pR);
        a_mean(kcfc) = mean(amp(indices));
        p_mean(kcfc) = mean([pL, pR]);
    end
    cfc_theta_cor_striatum = max(a_mean)-min(a_mean);


    
    phi = angle(hilbert(yy_striatum_THETA));
    amp = abs(hilbert(yy_cortex_THETA));
    p_bins = linspace(-pi,pi,63);
    a_mean = zeros(size(p_bins)-1);
    p_mean = zeros(size(p_bins)-1);
    for kcfc = 1:length(p_bins)-1
        pL = p_bins(kcfc);
        pR = p_bins(kcfc+1);
        indices=(phi>=pL) & (phi<pR);
        a_mean(kcfc) = mean(amp(indices));
        p_mean(kcfc) = mean([pL, pR]);
    end
    cfc_theta_striatum_cor = max(a_mean)-min(a_mean);    
    
    
end

