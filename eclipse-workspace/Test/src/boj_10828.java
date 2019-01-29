import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.util.Stack;
import java.util.StringTokenizer;

public class boj_10828 {
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		Scanner sc = new Scanner(System.in);
		Stack<Integer> stack = new Stack<>();

		int n = Integer.parseInt(br.readLine());
		for (int i = 0; i < n; i++) {
			StringTokenizer st = new StringTokenizer(br.readLine());
			String temp = st.nextToken();
			if (temp.contains("push")) {
				int k = Integer.parseInt(st.nextToken());
				stack.add(k);
			} else if (temp.contains("top")) {
				if (stack.size() == 0) {
					System.out.println("-1");
				} else {
					System.out.println(stack.peek());
				}
			} else if (temp.contains("size")) {
				System.out.println(stack.size());
			} else if (temp.contains("empty")) {
				if (stack.size() == 0) {
					System.out.println("1");
				} else {
					System.out.println("0");
				}
			} else if (temp.contains("pop")) {
				if (stack.size() == 0) {
					System.out.println("-1");
				} else {
					System.out.println(stack.pop());
				}

			}
		}

	}
}
