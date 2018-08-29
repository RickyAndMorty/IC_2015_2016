function [res] = fitness(P,Fth,K,Pmax)
wp = 1;
%Cost Function c)
%Pr = repmat(F',1,size(P,2)) .* P .* repmat(Gii',1,size(P,2));
res = (wp/K) .* sum(Fth .* (1 - (P ./ Pmax)),1);
%res = (ws./std(Pr,0,1)) + ws2./(K*var(SNR_tt)) + (wp/K) .* sum(Fth .* (1 - (P ./ Pmax)),1);