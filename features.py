import cv2
from pylab import *
from scipy.ndimage import filters


class FeatureDetector:
    def __init__(self, image):
        self.img = image

    def HarrisKeypointDetector(self, ratio, min):
        """
        :param ratio:Ratio to set the threshold
        :param min:If the neighborhood space distance is less than min, only one corner point is selected
        :return:The selected keypoints-coordinates
        """
        ksize = (3, 3)  # 3*3 or 5*5 kernel size
        gradient_imx, gradient_imy = np.zeros(self.img.shape), np.zeros(self.img.shape)
        # Return the partial derivatives in x and y
        filters.gaussian_filter(self.img, ksize, (0, 1), gradient_imx)
        filters.gaussian_filter(self.img, ksize, (1, 0), gradient_imy)

        # Caculate the elements in Harris Matrix
        Hx = cv2.GaussianBlur(gradient_imx * gradient_imx, ksize, sigmaX=0, sigmaY=0)
        Hy = cv2.GaussianBlur(gradient_imy * gradient_imy, ksize, sigmaX=0, sigmaY=0)
        Hxy = cv2.GaussianBlur(gradient_imx * gradient_imy, ksize, sigmaX=0, sigmaY=0)
        det = Hx * Hy - Hxy ** 2
        trace = Hx + Hy

        alpha = 0.04
        # Get the corner response
        resp = det - alpha * np.square(trace)
        threshold = ratio * resp.max()
        # Deep copy
        threshold_img = np.copy(resp)
        # Set the intensity of non-target points to zero
        threshold_img = (threshold_img > threshold) * 1
        # Get the coordinates of keypoints
        coords = np.array(threshold_img.nonzero()).T
        candidate_values = [self.img[c[0], c[1]] for c in coords]
        index = argsort(candidate_values)
        """
        Select the domain space.
        if the neighborhood space distance is less than min, only one corner point is selected
        """
        # Prevent the corners from being too dense
        neighbor = zeros(self.img.shape)
        neighbor[min:-min, min:-min] = 1
        filtered_coords = []
        for i in index:
            if neighbor[coords[i, 0], coords[i, 1]] == 1:
                filtered_coords.append(coords[i])
                neighbor[(coords[i, 0] - min):(coords[i, 0] + min),
                (coords[i, 1] - min):(coords[i, 1] + min)] = 0
        return filtered_coords


class FeatureDescriptor:
    def __init__(self, image, keypoints):
        self.img = image
        self.keypoints = keypoints

    def rotation_angles(self):
        ksize = (3, 3)
        gradient_imx, gradient_imy = np.zeros(self.img.shape), np.zeros(self.img.shape)
        filters.gaussian_filter(self.img, ksize, (0, 1), gradient_imx)
        filters.gaussian_filter(self.img, ksize, (1, 0), gradient_imy)

        # Caculate the elements in Harris Matrix
        Hx = cv2.GaussianBlur(gradient_imx * gradient_imx, ksize, sigmaX=0, sigmaY=0)
        Hy = cv2.GaussianBlur(gradient_imy * gradient_imy, ksize, sigmaX=0, sigmaY=0)
        Hxy = cv2.GaussianBlur(gradient_imx * gradient_imy, ksize, sigmaX=0, sigmaY=0)
        angles = []

        for i, f in enumerate(self.keypoints):
            x, y = f[0], f[1]
            mtrx = np.mat([[Hx[x][y], Hxy[x][y]], [Hxy[x][y], Hy[x][y]]])
            eigenvalues, eigenvectors = np.linalg.eig(mtrx)
            # find the larger eigenvalue and the corresponding eigenvector
            if eigenvalues[0] > eigenvalues[1]:
                index = 0
            else:
                index = 1
            # calculate the angle of principal axes
            angle = np.arctan(eigenvectors[1, index] / eigenvectors[0, index])
            angles.append([x, y, angle])

        return angles

    def SimpleFeatureDescriptor(self):
        img = self.img.astype(np.float32)
        img = np.pad(img, [(2, 2), (2, 2)], mode='constant')  # zero padding
        features = []

        # extract a small window (5*5) around each keypoint
        for i, f in enumerate(self.keypoints):
            row, col = f[0], f[1]
            row = row + 2
            col = col + 2
            window = img[row - 2:row + 3, col - 2:col + 3]
            features.append(window)

        return features

    def Simp_MOPS(self, angles):
        img = self.img.astype(np.float32)
        img = np.pad(img, [(35, 35), (35, 35)], mode='constant')  # zero padding
        features = []

        for i, f in enumerate(angles):
            row, col, angle = f[0], f[1], f[2]
            row = row + 35
            col = col + 35
            small_img = img[row - 35:row + 35, col - 35:col + 35]  # extract a small image (70*70) around each keypoint
            # Scale to 1/5 size (using prefiltering)
            small_img_filtered = filters.gaussian_filter(small_img, 5)
            small_img_scaling = cv2.resize(small_img_filtered, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            # Rotate to horizontal
            rot_mtrx = cv2.getRotationMatrix2D((7, 7), -angle, 1)
            small_img_scaling_rot = cv2.warpAffine(small_img_scaling, rot_mtrx, small_img_scaling.shape)
            # Sample 8x8 square window centered at feature
            window = small_img_scaling_rot[4:12, 4:12]
            # Intensity normalize the window by subtracting the mean, dividing by the standard deviation in the window
            window_mean = np.mean(window)
            window_std = np.std(window)
            for a in range(0, 8):
                for b in range(0, 8):
                    # If the variance is less than 10^-10 in magnitude, return 0 in window
                    if window_std < 10 ^ (-10):
                        window[a][b] = 0
                    else:
                        window[a][b] = (window[a][b] - window_mean) / window_std
            features.append(window)

        return features


class FeatureMatch:
    def __init__(self, feature1, feature2):
        self.ft1 = feature1
        self.ft2 = feature2

    def SSDFeatureMatcher(self):
        """
        Pass in features obtained from both images
        The data type received here is List
        Each element in the list is a 5*5 (simple) or 8*8 (MOPS) array
        """
        SSD = []  # Used to store the SSD distance between any two points in feature list
        SSDpairs = []  # Used to store the index values of the two matched elements in the two feature lists
        zerofeature1 = []  # Used to store index values of the elements that all values are 0 in the feature list
        nonzeroDes1 = []  # Used to store index value of the elements that not all values are 0
        for i in range(len(self.ft1)):
            if np.all(self.ft1[i] != 0):
                nonzeroDes1.append(i)
            else:
                zerofeature1.append(i)

        nonzeroDes2 = []
        zerofeature2 = []
        for j in range(len(self.ft2)):
            if np.all(self.ft2[j] != 0):
                nonzeroDes2.append(j)
            else:
                zerofeature2.append(j)

        for x in nonzeroDes1:
            for y in nonzeroDes2:
                SSD.append(np.sum((self.ft1[x] - self.ft2[y]) ** 2))
                # Perform SSD operations and add the values to the preset SSDlist
        SSD = np.array(SSD)
        SSD = np.resize(SSD, (len(nonzeroDes1), len(nonzeroDes2)))
        # Transform the number of rows and columns of the matrix
        # so that the SSD value corresponding to each point of interest can be obtained
        indexMatrix = np.argsort(SSD)  # Get the sort of SSD value size
        for x in range(len(nonzeroDes1)):
            SSDpairs.append([x, indexMatrix[x][0]])

        return SSD, SSDpairs, zerofeature1, zerofeature2, indexMatrix

    def RatioFeatureMatcher(self, threshold):
        ratiopairs = []  # Used to store the index values of the two matched elements in the two feature lists by using ratiotest
        SSD, SSDpairs, zerofeature1, zerofeature2, indexMatrix = FeatureMatch.SSDFeatureMatcher(self)
        ratio = np.zeros(((len(self.ft1) - len(zerofeature1)), 1),
                         dtype='float')  # Store the ratio calculation value of each matching point pair
        for t in range((len(self.ft1) - len(zerofeature1))):
            if SSD[t][indexMatrix[t][1]] == 0:  # Determines whether the second smallest point of the SSD value is 0
                ratio[t] = 1
            else:
                ratio[t] = SSD[t][indexMatrix[t][0]] / SSD[t][indexMatrix[t][1]]
        for m in range(len(SSDpairs)):
            if ratio[m] <= threshold:
                ratiopairs.append(
                    SSDpairs[m])  # Elements in ratiopAIRS are a subset of the number of elements in SSDpairs

        return np.array(ratiopairs), ratio


class FeaturePlot:
    def __init__(self, img1, img2, keypoints1, keypoints2, ratiopairs, zerofeature1, zerofeature2, SSD):
        self.img1 = img1
        self.img2 = img2
        self.kpt1 = keypoints1
        self.kpt2 = keypoints2
        self.ratiopairs = ratiopairs
        self.zeroft1 = zerofeature1
        self.zeroft2 = zerofeature2
        self.SSD = SSD

    def get_pos(self):
        nonzerokps1 = []
        nonzerokps2 = []
        position1 = []
        position2 = []
        for h in range(len(self.kpt1)):
            if h in self.zeroft1:
                continue
            else:
                nonzerokps1.append(list(self.kpt1[h]))

        for k in range(len(self.kpt2)):
            if k in self.zeroft2:
                continue
            else:
                nonzerokps2.append(list(self.kpt2[k]))

        nonzerokps1 = np.array(nonzerokps1)
        nonzerokps2 = np.array(nonzerokps2)
        # Eliminate individuals whose feature descriptors are all zero in points of interest
        IMG = concatenate((self.img1, self.img2), axis=1)
        nonzerokps2[:, 1] = nonzerokps2[:, 1] + self.img1.shape[1]  # Connect two images

        for i in range(len(self.ratiopairs)):
            position1.append(nonzerokps1[self.ratiopairs[i][0]])
            position2.append(nonzerokps2[self.ratiopairs[i][1]])  # Get the coordinates of the matched interest points
        position1 = np.array(position1)
        position2 = np.array(position2)
        return IMG, position1, position2, nonzerokps1, nonzerokps2

    def Connect_img(self):
        IMG, position1, position2, nonzerokps1, nonzerokps2 = FeaturePlot.get_pos(self)
        plt.figure()
        plt.axis('off')

        plt.plot([p[1] for p in nonzerokps1], [p[0] for p in nonzerokps1], '*')
        plt.plot([p[1] for p in nonzerokps2], [p[0] for p in nonzerokps2], '+')  # Delineate the feature points

        for x in range(len(self.ratiopairs)):
            IMG = cv2.line(IMG, (position1[x][1], position1[x][0]), (position2[x][1], position2[x][0]), (0, 255, 0), 2)
            # The matched points are connected
        plt.imshow(IMG)
        plt.show()
        return IMG

    def Plot_ROC(self, H, threshold, arg1, arg2):
        def ROC_data(IMG, coords1, coords2, threshold, Hmatrix):
            # Store coordinates obtained by matrix coordinate transformation
            mtr_H = []
            # Store the distance
            dists = []
            # The result of coordinate transformation
            match_res = []
            d = IMG.shape[1] / 2
            # Coords are filled in a column of 1 to enable them to perform matrix multiplication with 3*3 Harris matrix
            for i in range(len(coords1)):
                coords1[i][0], coords1[i][1] = coords1[i][1], coords1[i][0]
                coords2[i][0], coords2[i][1] = coords2[i][1] - d, coords2[i][0]
            z1 = np.ones(len(coords1))
            coords1_pad = np.c_[coords1, z1]
            z2 = np.ones(len(coords2))
            coords2_pad = np.c_[coords2, z2]
            for pos in coords1_pad:
                mtr_H.append(np.matmul(Hmatrix, pos))
            for i in range(len(coords2_pad)):
                dist = np.linalg.norm(mtr_H[i] - coords2_pad[i])
                dists.append(dist)
                if dist <= threshold:
                    # Match succeeds
                    match_res.append(1)
                else:
                    match_res.append(0)
            return match_res

        def computeROCCurve(matches, match_res, ssd_least_arr):
            dataPoints = []
            thresholds = np.linspace(arg1, arg2, 40)
            for threshold in thresholds:
                tp = 0
                actualCorrect = 0
                fp = 0
                actualError = 0
                total = 0

                for j in range(len(matches)):
                    if match_res[j]:
                        actualCorrect += 1
                        if ssd_least_arr[j] < threshold:
                            tp += 1
                    else:
                        actualError += 1
                        if ssd_least_arr[j] < threshold:
                            fp += 1

                    total += 1

                trueRate = (float(tp) / actualCorrect) if actualCorrect != 0 else 0
                falseRate = (float(fp) / actualError) if actualError != 0 else 0
                dataPoints.append([falseRate, trueRate])
                if [falseRate, trueRate] == [0, 1.0] or [falseRate, trueRate] == [1.0, 1.0]:
                    break
            dataPoints.insert(0, [0.0, 0.0])
            dataPoints.append([1.0, 1.0])
            return dataPoints

        def computeAUC(results):
            auc = 0

            for i in range(1, len(results)):
                falseRate, trueRate = results[i]
                falseRatePrev, trueRatePrev = results[i - 1]
                xdiff = falseRate - falseRatePrev
                ydiff = trueRate - trueRatePrev
                auc += xdiff * trueRatePrev + xdiff * ydiff / 2

            return auc

        IMG, position1, position2, nonzerokps1, nonzerokps2 = FeaturePlot.get_pos(self)
        match_res = ROC_data(IMG, position1, position2, threshold, Hmatrix=H)
        # ------------------------------------
        ssd_least_arr = []
        for x in range(len(self.ratiopairs)):
            ssd_least_arr.append(self.SSD[self.ratiopairs[x][0]][self.ratiopairs[x][1]])
        dataPoints = computeROCCurve(self.ratiopairs, match_res, ssd_least_arr)
        auc_value = computeAUC(dataPoints)
        return dataPoints, auc_value


# test
img1 = cv2.imread("yosemite1.jpg", 0)
img2 = cv2.imread("yosemite2.jpg", 0)
img1_color = cv2.imread("yosemite1.jpg")
img2_color = cv2.imread("yosemite2.jpg")
# BGR2RGB
img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

detector1 = FeatureDetector(img1)
detector2 = FeatureDetector(img2)
interesting_points1 = detector1.HarrisKeypointDetector(ratio=0.05, min=3)
interesting_points2 = detector2.HarrisKeypointDetector(ratio=0.05, min=3)

descriptor1 = FeatureDescriptor(img1, interesting_points1)
descriptor2 = FeatureDescriptor(img2, interesting_points2)
# ----------------------------------------------------
simple_features1 = descriptor1.SimpleFeatureDescriptor()
simple_features2 = descriptor2.SimpleFeatureDescriptor()
angles1 = descriptor1.rotation_angles()
mops_features1 = descriptor1.Simp_MOPS(angles1)
angles2 = descriptor2.rotation_angles()
mops_features2 = descriptor2.Simp_MOPS(angles2)
# -------------------------------------------------------
Matcher = FeatureMatch(mops_features1, mops_features2)
SSD, SSDpairs, zerofeature1, zerofeature2, indexMatrix = Matcher.SSDFeatureMatcher()
ratiopairs, ratio = Matcher.RatioFeatureMatcher(threshold=0.8)
ftplt = FeaturePlot(img1_color, img2_color, interesting_points1, interesting_points2, ratiopairs,
                    zerofeature1, zerofeature2, SSD)
# --------------------------------------------------------
Matcher_ = FeatureMatch(simple_features1, simple_features2)
SSD_, SSDpairs_, zerofeature1_, zerofeature2_, indexMatrix_ = Matcher_.SSDFeatureMatcher()
ratiopairs_, ratio_ = Matcher_.RatioFeatureMatcher(threshold=0.8)
ftplt_ = FeaturePlot(img1_color, img2_color, interesting_points1, interesting_points2, ratiopairs_,
                     zerofeature1_, zerofeature2_, SSD_)
ftplt.Connect_img()
ftplt_.Connect_img()
H = np.array([[1.065366, -0.001337, -299.163870],
              [0.027334, 1.046342, -11.093753],
              [0.000101, 0.000002, 1]])
dataPoints, auc_value = ftplt.Plot_ROC(H, 22, 0.4, 2.3)
print(f"AUC Value-MOPS:{auc_value}")
dataPoints_, auc_value_ = ftplt_.Plot_ROC(H, 22, 800, 10000)
print(f"AUC Value-Simple:{auc_value_}")
# Plot ROC
plt.figure()
plt.xlabel('Specificity(%)')
plt.ylabel('Sensitivity(%)')
plt.grid('on')
plt.title("ROC curve of the Model")
plt.plot([pts[0] for pts in dataPoints], [pts[1] for pts in dataPoints])
plt.plot([pts[0] for pts in dataPoints_], [pts[1] for pts in dataPoints_])
plt.show()
