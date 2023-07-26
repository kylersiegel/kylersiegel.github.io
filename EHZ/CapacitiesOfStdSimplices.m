maxDim = 10;
results = zeros(1,maxDim);
for n=1:maxDim
    stdSimplex = [eye(2*n); zeros(1,2*n)];
    results(n) = Capacity(stdSimplex,n);
end
