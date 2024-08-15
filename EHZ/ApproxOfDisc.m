iterations = 40;
results = zeros(1, iterations-2);

for numOfPoints=3:iterations
    points = exp(1).^(i*2*pi*[1:numOfPoints]/numOfPoints);
    X = real(points);
    Y = imag(points);
    P = [X;Y]';

    results(numOfPoints-2) = Capacity(P,1);
end

results
