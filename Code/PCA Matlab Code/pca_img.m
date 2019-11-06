%% Housekeeping

clear all;
close all;
clc;

VAR_FILTER = 95; % must be 90, 95, or 99
OFFSET = 0; % ignores the first X principal components

%% Read data

disp('Reading data...')
train = dlmread('../data/training.csv', ', ', 1, 0); % include a space in delimiter when in Octave
faces = train(:, 31:end); % each row is a face
[M,N] = size(faces);
disp('Done.')

%% Histogram stretch (increase contrast)

disp('Histogram stretching...')
histo_faces = faces;
for ii = 1:M
    histo_faces(ii, :) = histogram_stretch(faces(ii, :), 5, 95);
end
disp('Done.')

%% Scale so that each column (variable) has mean=0 and variace=1

disp('Scaling image data...')
histo_faces = zscore(histo_faces);
disp('Done.')

%% Eigen* decomposition

disp('Eigen* decomposition...')

S = cov(histo_faces); % 5 secs
eigvals = eig(S);% 72 secs
eigvals = eigvals(end:-1:1);

tv = pca_var(eigvals);
k = sum(tv < VAR_FILTER/100);
[V, D] = eigs(S, k);

V = V(:, 1+OFFSET:end);

disp('Done.')

%% Compute projections into PCA space

disp('Projecting into PCA space...')
X = (V' * histo_faces')';
histo_faces_pca = bsxfun(@plus, (V * X')', mean(histo_faces));
disp('Done.')

%% Visualize

for ii = 1:5
    subplot(3, 5, ii);
    imshow(reshape(rescale(faces(ii,:), 0, 255), 96, 96)', [0 255]);
    subplot(3, 5, ii+5);
    imshow(reshape(rescale(histo_faces(ii,:), 0, 255), 96, 96)', [0 255]);   
    subplot(3, 5, ii+10);
    imshow(reshape(rescale(histo_faces_pca(ii,:), 0, 255), 96, 96)', [0 255]);
end

%% Save for later use!!

disp('Saving...')
save eigenface.mat histo_faces histo_faces_pca
disp('Done.')