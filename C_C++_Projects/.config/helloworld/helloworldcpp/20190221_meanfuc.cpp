#include <iostream>
#include <stdio.h>
using namespace std;

int main(){
    int total;
    printf("전체 학생수 : ");
    scanf("%d", &total);
    int arr[total];
    int i, ave = 0;

    for(i=0; i<total; i++){
        printf("%d 첫번째 학생의 성적은?", i+1);
        scanf("%d", &arr[i]);
    }
    for(i=0; i<total; i++){
        ave = ave + arr[i];
    }
    
    ave = ave/total;
    printf("전체 학생의 평균은 : %d \n", ave);

    for(i=0; i<total; i++){
        printf("학생 %d : " , i + 1);
        if(arr[i] >= ave)
            cout << "합격" << endl;
        else
        {
            cout << "불합격" << endl;
        }
        
    }
}