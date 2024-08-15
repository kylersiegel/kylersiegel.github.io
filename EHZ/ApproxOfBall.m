n = 2;
results = zeros(6,1);
index = 1;
for p=[1.01 2 3 4 8 10]
    for numOfPoints = [60]
        points = 2*(0.5- rand(numOfPoints, 2*n));
        for i = 1:numOfPoints
            points(i,:)=points(i,:)/norm(points(i,:),p);
        end

        results(index) = Capacity(points,n);%, 'plotchar','on');
        index = index + 1;
    end
end
results

%{
grid on
axis square
axis([-1.5 1.5 -1.5 1.5])
%}
