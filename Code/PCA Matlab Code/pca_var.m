function tv = pca_var(eigvals)

eigsum = sum(eigvals);
csum = 0;
tv = zeros(1, length(eigvals));
for ii = 1:length(eigvals)
    csum = csum + eigvals(ii);
    tv(ii) = csum / eigsum;
end

end