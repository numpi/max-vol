function [I, Ares, traces] = cca2_spsd(A, r, debug, traces)
% Compute the indices I of the  r x r principal submatrices in the SPSD matrix A 
% which is guaranteed to be quasi-optimal in the nuclear norm error approximation
%
%
%---------------------------------------INPUT----------------------------------------------------
%
% A		    target matrix in full format  
% r 		size of the sought dominant submatrix
% debug		(optional) check the computation of the ratio of coefficients of poly(A)
% traces	(optional) vector containing the traces of A, A^2,...,A^(s+1) with s >= r
% 
%---------------------------------------OUTPUT---------------------------------------------------
%
% I			indices of the principal submatrix
% Ares		residual matrix A - A(:, I) / A(I, I) * A(:, I)
% traces	vector containing the traces of Ares, Ares^2,...,Ares^(s+1)
%
%------------------------------------------------------------------------------------------------
	if ~exist('debug', 'var')
		debug = 0;
	end
	
	not_taken = 1:size(A, 1);
	I = [];
	if debug
		fprintf('-----------------------------------------------\n')
		fprintf('Debug: Submatrix of size %d\n', r);
		pause
	end
	if ~exist('traces', 'var')
		s = r + 1;
		%S = eig(A);
		%traces = sum(S.^[1:r + 1], 1); % Compute the traces of A, A^2,...,A^(r+1)
		traces = compute_traces(A, r + 1);
	else
		s = length(traces);
		if s < r + 1
			error('CCA2_SPSD::The provided number of traces is not sufficient\n');
		end
	end
	for t = 1:r
		tmp_traces = zeros(length(not_taken), r - t + 2);
		d = sqrt(diag(A)).'; 
		d(not_taken) = 1 ./ d(not_taken);
		tmp_A = A .* d;
		U = tmp_A(:, not_taken);

		min_ratio = inf;
		min_ratio_ind = 0;
		if debug 
			min_true_ratio = inf;
			min_true_ind = 0;
		end
		if nargout == 3
			traces_corr = arnoldi_trace_powers_update(A, U, s);
			tmp_traces = traces_corr(:, 1:r - t + 2);
		else
			tmp_traces =  arnoldi_trace_powers_update(A, U, r - t + 2);
		end
		Tr  = toeplitz(traces(1:r - t + 1), [traces(1), zeros(1, r - t)]);        
        Tr1 = toeplitz(traces(1:r - t + 2), [traces(1), zeros(1, r - t + 1)]);

		for j = 1:length(not_taken)

			tmp_Tr  = Tr  \ (toeplitz(tmp_traces(j, 1:end-1), [tmp_traces(j, 1), zeros(1, r - t)]) + diag(r - t:-1:1, 1))    + eye(r - t + 1);
			tmp_Tr1 = Tr1 \ (toeplitz(tmp_traces(j, :),       [tmp_traces(j, 1), zeros(1, r - t + 1)]) + diag(r - t + 1:-1:1, 1)) + eye(r - t + 2);
			[~, U1, ~] = lu(tmp_Tr1); U1 = sort(abs(diag(U1)));
			[~, U2, ~] = lu(tmp_Tr);  U2 = sort(abs(diag(U2)));
			%U1 = sort(svd(tmp_Tr1));
			%U2 = sort(svd(tmp_Tr));

			tmp_ratio = abs(prod(U1(1:end-1) ./ U2) * U1(end));
			if  tmp_ratio < min_ratio
				min_ratio = tmp_ratio;
				min_ratio_ind = j;
			end
			if debug
				p = poly(A - U(:, j) * U(:, j)'); % characteristic polynomial of the residual matrix
				true_ratio = abs(p(r - t + 3) / p(r - t + 2));
				if  true_ratio < min_true_ratio
					min_true_ratio = true_ratio;
					min_true_ratio_ind = j;
				end	
			end
		end
		if debug
				p = poly(A - U(:, min_ratio_ind) * U(:, min_ratio_ind)'); 
				tmp_ratio = abs(p(r - t + 3) / p(r - t + 2));
				err = abs((tmp_ratio - min_true_ratio)/min_true_ratio);
				fprintf('t = %d, Selected index = %d, True minimum = %d, relative error = %1.2e, sel. val.: %1.2e, true val.: %1.2e \n',...
				 t, not_taken(min_ratio_ind), not_taken(min_true_ratio_ind), err, tmp_ratio, min_true_ratio);
		end
		A = A - tmp_A(:, not_taken(min_ratio_ind)) * tmp_A(:, not_taken(min_ratio_ind))'; % update A
		A(:, not_taken(min_ratio_ind)) = 0; A(not_taken(min_ratio_ind), :) = 0; 		  % enforce zeros on the cross
		I = [I, not_taken(min_ratio_ind)];  											  % append index to I
		if nargout == 3 
			traces = traces + traces_corr(min_ratio_ind, :);
		else 
			traces = traces(1:end - 1) + tmp_traces(min_ratio_ind, 1:end - 1); 			  % update powers' traces  
		end
		not_taken = not_taken([1:min_ratio_ind - 1, min_ratio_ind+1:length(not_taken)]);  % update not_taken
	end
	I = sort(I);
	Ares = A;
end

function tr = compute_traces(A, r, Aold)
	if ~exist('Aold', 'var')
		Aold = A;
	else
		Aold = A * Aold;
	end
	if r > 1
		tr = [trace(Aold), compute_traces(A, r - 1, Aold)];
	else
		tr = trace(Aold);
	end
end


