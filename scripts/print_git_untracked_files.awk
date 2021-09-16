BEGIN {
    untracked_matched = 0
}

{
    if ($0 ~ /^Untracked files:/) {
	untracked_matched = 1
    }

    if (untracked_matched) {
	print $0
    }
}
