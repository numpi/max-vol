function [I, Ares, traces] = quasi_cca_spsd(A, r, t, debug)
% Compute the indices I of the  rank r cross approximation of the SPSD matrix A 
% by restarting every t iterations the method that is  guaranteed to be quasi-optimal 
% in the nuclear norm error approximation
%
%
%---------------------------------------INPUT----------------------------------------------------
%
% A		    target matrix in full format  
% r 		size of the sought dominant submatrix
% t			max number of iterations before restarting
% debug		activate the debugging prints in cca2_spsd.m
% 
%---------------------------------------OUTPUT---------------------------------------------------
%
% I			indices of the principal submatrix
% Ares		residual matrix A - A(:, I) / A(I, I) * A(:, I)
% traces	vector containing the traces of Ares, Ares^2,...,Ares^(r+1)
%
%------------------------------------------------------------------------------------------------
	if r <= 0
		error('QUASI_CCA_SPSD:: parameter r must be positive')
	end
	if ~exist('debug', 'var')
		debug = 0;
	end
	I = [];
	not_taken = 1:size(A, 1);
	tmp_not_taken = not_taken;
	while r > t
		[II, A] = cca2_spsd(A, t);
		I = [I, not_taken(II)];
		not_taken = setdiff(not_taken, not_taken(II));
		tmp_not_taken = setdiff(1:size(A, 1), II);
		A = A(tmp_not_taken, tmp_not_taken);
		r = r - t;
	end
	if r > 0
		[II, A] = cca2_spsd(A, r);
		I = [I, not_taken(II)];
	end
	I = sort(I);
	Ares = A;
end
