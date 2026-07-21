#ifndef PARTITION_HPP
#define PARTITION_HPP

class Partition
{
private:
    int n;
    int rank;
    int size;

    int first_row;
    int local_rows;

public:
    Partition(int n_value, int rank_value, int size_value);

    int get_first_row() const;
    int get_last_row() const;
    int get_local_rows() const;
    int get_local_size() const;
};

#endif
