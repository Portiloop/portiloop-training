function [ratio] = compute_ratio(idx, signal, L, f28, f916)
X = signal(idx-L+1:idx, 1);
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:ceil(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
deltatheta = mean(P1(f28));
spin = mean(P1(f916));
ratio = spin/deltatheta;
end

