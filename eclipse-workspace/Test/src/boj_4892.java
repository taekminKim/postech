import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class boj_4892 {
	public static void main(String[] args) throws NumberFormatException, IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int cnt = 1;
		while (true) {
			int n0 = Integer.parseInt(br.readLine());
			int n1 = 0;
			int n2 = 0;
			int n3 = 0;
			int n4 = 0;
			if (n0 == 0) {
				break;
			}
			n1 = n0 * 3;
			if (n1 % 2 == 0) {
				// n1이 짝수라면
				n2 = n1 / 2;
				n3 = n2 * 3;
				n4 = n3 / 9;
				System.out.println(cnt + ". even " + n4);
			} else {
				n2 = (n1 + 1) / 2;
				n3 = n2 * 3;
				n4 = n3 / 9;
				System.out.println(cnt +". odd " +n4);
			}
			cnt++;
		}
	}
}
