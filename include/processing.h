
#include <algorithm>

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

inline int mod(int a, int b)
{
    int ret = a % b;
    if (ret<0)
        ret+=b;
    return ret;
}

template<typename T> double meanOf(const T& array){
    double mean = 0;
    for (auto& val: array){
        mean+=val;
    }
    return mean/((double)array.size());
}

template<typename T> double variance(const T& array){
    double mean = meanOf(array);
    double variance = 0.0;
    for(auto& val : array){
        auto diff = val - mean;
        variance += diff * diff;
    }
    return variance/((double)array.size());
}