def execute(in_map):
   if in_map['day_time']>74070.7:
      if in_map['day_time']>88385.7:
         if in_map['clouds']>3:
            if in_map['clouds']>4:
               if in_map['day_time']>93655.1:
                  return  1
               else:
                  return  0
            else:
               if in_map['day_time']>91894.6:
                  if in_map['day_time']>93947.4:
                     return  0
                  else:
                     return  1
               else:
                  return  0
         else:
            if in_map['day_time']>89118.4:
               if in_map['day_time']>90270:
                  return  1
               else:
                  return  0
            else:
               return  1
      else:
         if in_map['clouds']>2:
            if in_map['day_time']>83495.4:
               if in_map['day_time']>85353.8:
                  if in_map['clouds']>3:
                     return  1
                  else:
                     return  0
               else:
                  if in_map['day_time']>84494.5:
                     return  0
                  else:
                     return  1
            else:
               if in_map['day_time']>82616.2:
                  if in_map['clouds']>4:
                     return  0
                  else:
                     return  1
               else:
                  if in_map['day_time']>82075.5:
                     return  0
                  else:
                     return  1
         else:
            if in_map['day_time']>74892.1:
               return  1
            else:
               return  0
   else:
      if in_map['day_time']>65736.9:
         if in_map['clouds']>2:
            if in_map['day_time']>71234.8:
               if in_map['day_time']>72774.6:
                  if in_map['day_time']>72959.2:
                     return  0
                  else:
                     return  1
               else:
                  if in_map['day_time']>72243.8:
                     return  0
                  else:
                     return  1
            else:
               if in_map['day_time']>68463.6:
                  if in_map['day_time']>70768:
                     return  0
                  else:
                     return  1
               else:
                  if in_map['day_time']>68445.5:
                     return  0
                  else:
                     return  1
         else:
            if in_map['day_time']>69131.1:
               if in_map['day_time']>69997.1:
                  return  1
               else:
                  return  0
            else:
               if in_map['clouds']>1:
                  if in_map['day_time']>67276.8:
                     return  1
                  else:
                     return  0
               else:
                  if in_map['day_time']>66286.5:
                     return  1
                  else:
                     return  0
      else:
         if in_map['clouds']>3:
            if in_map['day_time']>51399.2:
               if in_map['day_time']>57807.6:
                  if in_map['day_time']>58652.6:
                     return  1
                  else:
                     return  0
               else:
                  if in_map['clouds']>4:
                     return  1
                  else:
                     return  0
            else:
               return  1
         else:
            if in_map['day_time']>59469.9:
               if in_map['clouds']>2:
                  if in_map['day_time']>61794.8:
                     return  0
                  else:
                     return  1
               else:
                  if in_map['day_time']>61278.2:
                     return  1
                  else:
                     return  0
            else:
               if in_map['day_time']>56960.6:
                  return  1
               else:
                  if in_map['day_time']>56533.9:
                     return  0
                  else:
                     return  1
