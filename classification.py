


def chi_square(source,target):


  total=source+target
  s=sum(source)
  t=sum(target)
  total_sum=s+t


  Scontibution=s/total_sum
  Tcontibution=t/total_sum
  fe1=total*Scontibution
  fe2=total*Tcontibution
  result=[]
  for i in range(len(source)):
      x=source[i]-fe1[i]
      result.append((x**2)/fe1[i])

  for i in range(len(target)):
      x=target[i]-fe2[i]
      result.append((x**2)/fe2[i])
  chi_square_value=sum(result)

  return chi_square_value