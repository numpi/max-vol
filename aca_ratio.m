function [Ind, det_ratio, errs] = aca_ratio(A, B, n, k, tol, debug, isAspsd)
% Adaptive cross approximation for maximizing abs( det(A(Ind, Ind)) / det(B(Ind, Ind)) )
% with B SPD. The matrices can be provided either as handle functions of indices or in the full format
%
%----------------------------------------------------------------------------------------------------------------------------------
%
% A, B 		can be either the matrices in full format or handle functions that extract submatrices (e.g.  A(I, J))
% n			dimension of the matrices, optional if either A or B are given as full matrices
% k			cardinality of the sought Ind
% tol		(optional) threshold for the stopping trace criterion
% debug		(optional) prints the trace residuals and check presence of negative diagonal entries in the Schur complement
% isAspsd  	(optional) indicates whether A is SPSD
%
%---------------------------------------------------------------------------------------------------------------------------------- 
	if ~exist('n', 'var') || isempty(n)
		if isfloat(A)
			n = size(A, 1);
		else
			error('ACA_RATIO:: unspecified dimension of the matrix')
		end
	end	
	if ~exist('k', 'var') || k > n
		k = min(n, 5);
	end
	if ~exist('tol', 'var')
		tol = 1e-12;
	end
	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('isAspsd', 'var')
		isAspsd = 0;
	end
	
	errs = [];
	res = inf;
	det_ratio = 1;
	UA = zeros(n, 0); 
	if ~isAspsd
		VA = zeros(n, 0);
	end
	UB = zeros(n, 0);
	Ind = [];
	if debug
		A0 = A; B0 = B;
	end

	% Precompute diagonal entires of A and B
	if isfloat(A) 
		dA = diag(A);
	else
		dA = zeros(n, 1); 
		for j = 1:n
			dA(j) = A(j, j);
		end
	end
	if isfloat(B) 
		dB = diag(B);
	else
		dB = zeros(n, 1); 
		for j = 1:n
			dB(j) = B(j, j);
		end
	end

	it = 1;
	while res > tol && it <= k	

		% Compute new pivot
		dd = abs(dA ./ dB);
		dd(Ind) = -inf;    % prevents the choice of the pivots with indices in Ind
		[~, jmax] = max(dd); 
		det_ratio = det_ratio * dA(jmax) / dB(jmax);

		% ACA step on A
		if ~isAspsd
			tmp = A([1:n], jmax) - UA * VA(jmax, :)';
			UA = [UA, tmp / tmp(jmax)];
			VA = [VA, A(jmax, [1:n])'];
			UA(Ind, end) = 0; VA(Ind, end) = 0;     % (optional) enforce the exactness property on the submatrix already covered by the cross
			dA = dA - UA(:, end) .* VA(:, end);     % compute diagonal entries of the residual (Schur complement)
		else
			tmp = A([1:n], jmax) - UA * UA(jmax, :)';
			UA = [UA, tmp / sqrt(tmp(jmax))];
			UA(Ind, end) = 0;
			dA = dA - UA(:, end).^2;
		end

		% ACA step on B
		tmp = B([1:n], jmax) - UB * UB(jmax, :)';
		UB = [UB, tmp / sqrt(tmp(jmax))];
		UB(Ind, end) = 0;        % (optional) enforce the exactness property on the submatrix covered by the cross
		dB = dB - UB(:, end).^2; % compute diagonal entries of the residual (Schur complement)	

		Ind = [Ind, jmax];
		res = sum(dB);           % trace of the Schur complement of B (it bounds ||B - UB * UB'||_2)
		errs = [errs; res];      % Stores residues

		if debug
			fprintf('It= %d, det_ratio = %1.2e, true det_ratio = %1.2e,  Res = %1.2e\n', it, det_ratio, det(A0(Ind, Ind))/det(B0(Ind, Ind)), res);
			if sum(dB < 0) > 0
				fprintf('Negative diagonal entries in the Schur compl.: %d, Minimal diagonal entry = %1.2e \n', sum(dB < 0), min(dB));
			end
		end
		it = it + 1;
	end
end
