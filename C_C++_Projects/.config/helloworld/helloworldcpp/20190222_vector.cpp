#include <iostream>
#include <vector>
using namespace std;

int main(){
    vector<int> vec;
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);
    vec.push_back(40);

    for(vector<int>::size_type i  = 0; i<vec.size(); i++){
        cout << "vec의 " << i + 1 << " 번째 원소 :: " << vec[i] << endl;
    }

    //전체 벡터를 출력하기
    for(vector<int>::iterator itr = vec.begin(); itr != vec.end(); itr++){
        cout << *itr << endl;
    }
    // int arr[4] = {10,20,30,40}
    // *(arr + 2) == arr[2] == 30;
    // *(itr + 2) == vec[2] == 30;

    vector<int>::iterator itr = vec.begin() + 2;
    cout << "3 번째 원소 :: " << *itr << endl;
}