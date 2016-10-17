plus_primeR <- function(num){
  if(num%%(num^0.5) == 0){
    print(FALSE)
    }else if(num == 2){
      print(TRUE)
      }else if(num <2){
        print(FALSE)
      }else if (num == 3){
        print(TRUE)
      }else if (num%%3 == 0){
        print(FALSE)
      }else if (num%%8 == 0){
        print(FALSE)
        } else if(num%%(num^0.5) > 0){
          print(TRUE)
        }
      }
    
  
