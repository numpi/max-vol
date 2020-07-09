function [U, Ind, errs] = aca_spsd(Afun, n, tol, max_it, debug, taken, d)
% Adaptive cross approximation for symmetric positive definite matrix
% provided either as an handle function of indices or in the full format
%
%----------------------------------------------INPUT-------------------------------------------------------------------------------
%
% Afun 		can be either the matrix in full format or a handle function that extracts submatrices (e.g.  Afun(I, J))
% n			dimension of the matrix, optional if Afun is given as full matrix
% tol		(optional) threshold for the stopping trace criterion
% max_it	(optional) maximum number of iterations
% debug		(optional) prints the trace residuals and check presence of negative diagonal entries in the Schur complement
% taken		(optional) prevents the method to choose pivots in the cols specified in this variable (useful for aca_spsd_param)
% d        	(optional) diagonal entries of A (useful if precomputed)
%
%----------------------------------------------OUTPUT------------------------------------------------------------------------------
%
% U 		Low-rank Cholesky factor Afun = U * U'
% Ind		row (and column) index of the cross approximation
% errs		Sequence of trace errors 
%
%----------------------------------------------------------------------------------------------------------------------------------
	if ~exist('n', 'var')
		if isfloat(Afun)
			n = size(Afun, 1);
		else
			error('ACA_SPSD:: unspecified dimension of the matrix')
		end
	end	
	if ~exist('tol', 'var')
		tol = 1e-12;
	end
	if ~exist('max_it', 'var') || max_it > n
		max_it = n;
	end
	if ~exist('debug', 'var')
		debug = 0;
	end
	if ~exist('taken', 'var')
		taken = [];
	end
	
	errs = [];
	res = inf;
	it = 1;
	U = zeros(n, 0);
	Ind = [];
	if ~exist('d', 'var')
		if isfloat(Afun) % store diagonal entries in d
			d = diag(Afun);
		else
			d = zeros(n, 1); 
			for j = 1:n
				d(j) = Afun(j, j);
			end
		end
	end
	if debug
		fprintf('Number of negative diagonal entries in A: %d\n', sum(d<0));
	end
	while res > tol && it <= max_it	
		dd = d;
		dd(taken) = -inf;    % prevents the choice the of the pivots with indices in taken
		[~, jmax] = max(dd); % choose new pivot
		Ind = [Ind, jmax];
		tmp = Afun([1:n], jmax) - U * U(jmax, :)';
		U = [U, tmp / sqrt(tmp(jmax))];
		U(taken, end) = 0; % enforce the exactness property on the submatrix covered by the cross
		taken = [taken, jmax];	
		%compute diagonal entries of the residue
		d = d - abs(U(:, end)).^2;

		res = sum(d); % trace of Schur complement
		errs = [errs; res]; % Stores residues
		if debug
			fprintf('It: %d, Res: %e, Negative diagonal entries in the Schur compl.: %d\n', it, res, sum(d<0));
			if sum(d<0) > 0
				fprintf('Minimal diagonal entry = %e \n', min(d));
			end
		end
		it = it + 1;
	end
end
