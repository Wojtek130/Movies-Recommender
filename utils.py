import re

def find_movie(df, identifier):
    if isinstance(identifier, int):  # Search by 'id'
        result = df[df['id'] == identifier]
        if result.shape[0] == 0:
            print(f"No film found with id {identifier}.")
            return None
        return result

    elif isinstance(identifier, str):  # Search by 'title'
        # First, check for an exact match
        result_exact = df[df['title'] == identifier]
        
        if result_exact.shape[0] == 1:
            # Exact match found
            return result_exact
        if result_exact.shape[0] > 1:
            # Multiple exact matches
            print("Warning: Multiple films with the exact title. Please provide a unique title or id.")
            print("Found exact matches:")
            print(result_exact[['id', 'title']])  # Display all matching titles
            return None
        else:
            # No exact match, so proceed with regex search
            result_regex = df[df['title'].str.contains(identifier, flags=re.IGNORECASE, regex=True)]
            
            if result_regex.shape[0] > 1:
                print("Warning: Multiple films match the title pattern. Please provide a unique title or id.")
                print("Found matching titles:")
                print(result_regex[['id', 'title']])  # Display all matching titles
                return None
            if result_regex.shape[0] == 0:
                print(f"No films found with the title matching '{identifier}'.")
                return None
            
            return result_regex
    else:
        raise ValueError("Identifier must be either an integer (id) or a string (title)")