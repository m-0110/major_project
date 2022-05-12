
import numpy as np
import features_extract
import face_detector

#for each directional kernel create an object with operator as attribute
class kernel:
  def __init__(self,m):
      self.M=m

#kernel objects
M0= np.array([
  [-3 , -3 ,5],
  [-3 , 0 , 5],
  [-3, -3 , 5]
])
k0=kernel(M0)
M1=np.array([
  [-3 , 5 ,5],
  [-3 , 0 , 5],
  [-3, -3 ,-3]
])
k1=kernel(M1)

M2 = np.array([
  [5, 5, 5],
  [-3, 0, -3],
  [-3, -3, -3]
])
k2=kernel(M2)
M3 = np.array([
  [5, 5, -3],
  [5, 0, -3],
  [-3, -3, -3]
])
k3=kernel(M3)
M4 = np.array([
  [5, -3, -3],
  [5, 0, -3],
  [5, -3, -3]
])
k4=kernel(M4)
M5 = np.array([
  [-3, -3, -3],
  [5, 0, -3],
  [5, 5, -3]
])
k5=kernel(M5)
M6 = np.array([
  [-3, -3, -3],
  [-3, 0, -3],
  [5, 5, 5]
])
k6=kernel(M6)

M7 = np.array([
  [-3, -3, -3],
  [-3, 0, 5],
  [-3, 5, 5]
])
k7=kernel(M7)


class Image:
  def __init__(self, matrix):
    self.IM = matrix




#list of objects
k=[k0, k1,k2, k3, k4, k5, k6, k7]

def anova_test(source,target):

  n1=len(source)
  n2=len(target)
  k=2

  def variance(a):
      x=np.mean(a)
      n=len(a)
      s=0
      for i in a:
          s=s+((i-x)**2)
      return (s/(n-1))


  x1=np.mean(source)
  x2=np.mean(target)
  x=(n1*x1+n2*x2)/(n1+n2) #grand mean
  between_class_variance=( (  n1 *( (x1-x)**2 )  )  +  (  n2 *( (x2-x)**2)  ) )/(k-1)
  #print(between_class_variance)
  #print(x1,x2,x)
  ntotal=n1+n2
  var1=variance(source)
  var2=variance(target)
  #print(var1,var2)
  #within_class_variance=round((( (  (n1-1) / (ntotal-k) ) *var1 ) + ( (  (n2-1) / (ntotal-k) ) *var2 )),3)
  within_class_variance=(( (  (n1-1) / (ntotal-k) ) *var1 ) + ( (  (n2-1) / (ntotal-k) ) *var2 ))
  #print(within_class_variance)
  f_ratio=between_class_variance/within_class_variance
  #print(f_ratio)
  return abs(f_ratio)

# convert to decimal
def getEncodedValue(values):
  s=0
  for i in range(8):
      s=s+values[i]*(2**i)
  return s

#encode to 0 and 1
def encoding(m,values):
    r=[]
    for i in values:
        if(i>m): # if greater than m edge response becomes to 1 else 0
            r.append(1)
        else:
            r.append(0)
    return r


def superimpose(matrix, k, d):
  values = []

  # consider the operator m0 -m7
  operator = k[d]

  s = 0
  # calculate the egde response for the center pixel
  for i in range(3):
    for j in range(3):
      s = s + matrix[i][j] * operator.M[i][j]

  # not bothered abt the direction so absolute value of edge response is considered
  return abs(s)


def compare_img(imgs):#imgs have I-M0,...
  n=imgs[0].IM.shape[0]
  #print(n)
  resultant_image=[]

  #pick 3x3 part of image I-M(i) & I-M(i+1)
  for i in range(n-3+1):
    for j in range(n-3+1):
      result=[]
      r=0
      d=0
      while r<=7:

        obj=imgs[d]
        I1=obj.IM

        d=(d+1)%8
        obj=imgs[d]
        I2=obj.IM

        matrix1=I1[i:i+3,j:j+3]
        matrix2=I2[i:i+3,j:j+3]

        #reshaping 2d array of 3x3 into 1x 9 (with on ) and pick 1st column
        source=matrix1.reshape(1,9)[0]
        target=matrix2.reshape(1,9)[0]

        x_val=anova_test(source,target)
        result.append(x_val)

        r+=1
      m=sum(result)//8
      encodedvals = encoding(m, result)
      val = getEncodedValue(encodedvals)
      resultant_image.append(val)
  resultant_image=np.array(resultant_image)
  resultant_image=resultant_image.reshape(n-2,n-2)
  return resultant_image


# here we get I-m0 I-m1 I-m2...I-m7

def feature_extraction(img, file_name):
  l, n = face_detector.face_detection(img, file_name)

  # print("l",l)
  result = []
  d = 0
  # pick one by one operator
  I_M = []
  while d <= 7:
    values = []

    # first d=0
    for i in range(n - 3 + 1):
      for j in range(n - 3 + 1):
        # considering 3x3 part of img
        matrix = l[i:i + 3, j:j + 3]
        # print(matrix)

        val = superimpose(matrix, k, d)  # superimpose first d=0 for every 3 x3 matrix of I
        values.append(val)

    # after superimpose of one operator
    values = np.array(values)
    values = values.reshape(n - 2, n - 2)
    # print('Values',values)
    # print('values', values)
    obj = Image(values)
    I_M.append(obj)

    d += 1  # take next operator

  result = compare_img(I_M)

  features = features_extract.pca(result)
  # print(result)
  return features
  # return []
