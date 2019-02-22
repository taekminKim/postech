#include <iostream>
#include <stdio.h>
using namespace std;

int main(){
    int arr[5]; //성적을 저장하는 배열
    int i, ave = 0;

    for(i = 0; i<5; i++){
        //학생들의 성적을 이력받는 부분
        printf("%d 번째 학생의 성적은? ", i+1);
        scanf("%d", &arr[i]);
    }
    //전체 학생 성적의 합을 구하는 부분
    for(i=0; i<5; i++){
        ave= ave + arr[i];
    }

    printf("전체 학생의 평균은 : %d \n", ave/5 );
    //평균이므로 5로 나누어준다.
    return 0;

}