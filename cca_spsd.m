function [I, Ares] = cca_spsd_true(A, r, debug)
% Compute the indices I of the  r x r principal submatrices in the SPSD matrix A 
% which is guaranteed (certified) to be quasi-optimal in the nuclear norm error approximation
%
%
%---------------------------------------INPUT----------------------------------------------------
%
% A		    target matrix in full format  
% r 		size of the sought dominant submatrix
% debug		(optional) check the computation of the ratio between coefficients of poly(A)
% 
%---------------------------------------OUTPUT---------------------------------------------------
%
% I			indices of the principal submatrix
% Ares		residual matrix A - A(:, I) / A(I, I) * A(:, I)
%
%------------------------------------------------------------------------------------------------
	if ~exist('debug', 'var')
		debug = 0;
	end

	% Compute the ratio between coefficients of the characteristic polynomial of A

	not_taken = 1:size(A, 1);
	I = [];
	for t = 1:r
		d = sqrt(diag(A)).'; 
		d(not_taken) = 1 ./ d(not_taken);
		tmp_A = A .* d;
		U = tmp_A(:, not_taken);
		min_ratio = inf;
		min_ratio_ind = 0;

		[V, D] = eig(A);
		for j = 1:length(not_taken)
			tmp = V' * U(:, j);
			%B = D - tmp * tmp';
			%pol = poly(B);
			T = eigenv2(D, tmp);
			T = tril(triu(T, -1), 1);
			T = (T + T')/2;
			l = eig(T);
			pol = poly(l); 
			% Compute ratio
			ratio = abs(pol(r - t + 3) / pol(r - t + 2));
			if  ratio < min_ratio
				min_ratio = ratio;
				min_ratio_ind = j;
			end
		end

		% Select the minimum and update 
		A = A - tmp_A(:, not_taken(min_ratio_ind)) * tmp_A(:, not_taken(min_ratio_ind))'; % update A
		A = (A + A')/2;
		I = [I, not_taken(min_ratio_ind)];  											  % append index to I
		not_taken = not_taken([1:min_ratio_ind - 1, min_ratio_ind+1:length(not_taken)]);  % update not_taken
	end
	I = sort(I);
	Ares = A;
end



