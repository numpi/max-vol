function [V, H, params, lucky] = poly_krylov_sim(varargin)
%EK_KRYLOV Extended Krylov projection of a matrix A.
%
% [V, K, H, params] = POLY_KRYLOV(A, B) construct the extended Krylov
%     subspace spanned by [B, A*B, ...]. The matrix V is an orthogonal
%     basis for this space, and K and H are block upper Hessenberg
%     rectangular matrices satisfying
%
%        A * V * K = V * H                                          (1)
%
% [V, K, H, params] = POLY_KRYLOV(V, K, H, PARAMS) enlarges a
%     Krylov subspace generated with a previous call to POLY_KRYLOV by adding
%     infinity pole pair. The resulting space will satisfy
%     the same relation (1).
%
% Note: lucky is a flag parameter that highlight the lucky breakdown of lanczos
%
	if nargin ~= 2 && nargin ~= 3
		error('Called with the wrong number of arguments');
	end

	if nargin == 2
		% Start to construct the extended Krylov space
		[V, H, params, lucky] = poly_krylov_start_sim(varargin{:});
	else
		% Enlarge the space that was previously built
		[V, H, params, lucky] = poly_krylov_extend_sim(varargin{:});
	end
end
%-------------------------------------------------------------------------
function [V, H, params, lucky] = poly_krylov_start_sim(A, b)

	n = size(b, 1);

	% Construct a basis for the column span of b
	t = size(b, 2);
	V = zeros(n, t);
	V = b ./ vecnorm(b);

	H = {};
	for j = 1:t
		H = [H, {zeros(1, 0)}];
	end

	[V, H, w, lucky] = add_inf_pole_sim (mat2cell(V, n, ones(1, t)), H, A, V);

	% Save parameters for the next call
	params = struct();
	params.last = w;
	params.A = A;
end
%--------------------------------------------------------------------------
function [V, H, params, lucky] = poly_krylov_extend_sim(V, H, params)
	w  = params.last;
	A  = params.A;
	[V, H, w, lucky] = add_inf_pole_sim (V, H, A, w);
	params.last = w;
end

%----------------------------------------------------------------------------
% Utility routine that adds an infinity pole to the space. The vector w is
% the continuation vector.
%
function [V, H, w, lucky] = add_inf_pole_sim(V, H, A, w)
	lucky_tol = eps; % tolerance for detecting lucky breakdowns
	lucky = false;
	t = size(w, 2);

	if isstruct(A)
		w = A.multiply(1.0, 0.0, w);
	else
		w = A * w;
	end

	for j = 1:t
		% Enlarge H 
		H{j}(size(H{j}, 1) + 1, size(H{j}, 2) + 1) = 0;

		% Perform orthogonalization with modified Gram-Schimidt
		[w(:, j), H{j}(1:end - 1, end)] = mgs_orthogonalize(V{j}, w(:, j));
		nrm = norm(w(:, j));
		w(:, j) = w(:, j)/nrm;
		H{j}(end, end) = nrm;
		if nrm < lucky_tol %&& false
			lucky = true;
		end
		V{j} = [V{j}, w(:, j)];
	end
end

%
% Modified Gram-Schmidt orthogonalization procedure.
%
% Suggested improvements: work with block-size matrix vector products to
% get BLAS3 speeds.
%
function [w, h] = mgs_orthogonalize(V, w)
    h = V' * w;
    w = w - V * h;
    h1 = V' * w;
    h = h + h1;
    w = w - V * h1;
end

