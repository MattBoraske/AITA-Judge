# main.py
import dc_script
import df_script
import script3

def create_AITA_dataset():
    print("Running raw datafile filtering script...")
    df_script.main()  # Assuming each script has a main() function
    
    print("Creating dataset from refined datafile...")
    dc_script.main()

if __name__ == "__main__":
    create_AITA_dataset()