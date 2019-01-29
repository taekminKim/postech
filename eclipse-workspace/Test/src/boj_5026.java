import java.util.Scanner;

public class boj_5026 {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int n = sc.nextInt();
		String temp = sc.nextLine();
		System.out.println(temp.split("+"));
		/*
		for(int i=0; i<n; i++) {
			String temp = sc.nextLine();
			if(temp.contains("NP")) {
				System.out.println("skipped");
			}else {
				System.out.println(temp.split());
			}
		}
		*/
	}
}
