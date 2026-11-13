

#ifndef dynamic_bitset_h
#define dynamic_bitset_h

#include <iostream>
#include <string>
#include <stdexcept>
#include <exception>
#include <vector>
#include <iostream>
#include "integer.h"

#ifndef FAST_CODE
#define FAST_CODE use
#endif

namespace whiteice
{
  class dynamic_bitset
  {
  public:
    
    inline dynamic_bitset();
    inline dynamic_bitset(const dynamic_bitset& bitset);
    inline dynamic_bitset(unsigned long val);
    inline dynamic_bitset(const whiteice::math::integer& i);
    
    inline virtual ~dynamic_bitset();
    
    inline dynamic_bitset& operator=(const dynamic_bitset& bitset);
    
    inline dynamic_bitset& operator&=(const dynamic_bitset& r);
    inline dynamic_bitset& operator|=(const dynamic_bitset& r);
    inline dynamic_bitset& operator^=(const dynamic_bitset& r);
    inline dynamic_bitset& operator<<=(int pos) ;
    inline dynamic_bitset& operator>>=(int pos) ;
    
    // adds bitset at the beginning of the current one
    inline dynamic_bitset& operator+=(const dynamic_bitset& r);
    
    inline dynamic_bitset operator&(const dynamic_bitset& r) const;
    inline dynamic_bitset operator|(const dynamic_bitset& r) const;
    inline dynamic_bitset operator^(const dynamic_bitset& r) const;
    inline dynamic_bitset operator<<(const int pos) const ;
    inline dynamic_bitset operator>>(const int pos) const ;
    
    // ORs bit sequencies
    inline dynamic_bitset operator+(const dynamic_bitset& r) const;
    
    inline dynamic_bitset& set() ; // sets all bits
    inline dynamic_bitset& set(unsigned int pos, bool val = true) ;
    
    inline dynamic_bitset& reset() ; // clears all bits
    inline dynamic_bitset& reset(unsigned int pos) ;
    
    inline dynamic_bitset& flip() ;
    inline dynamic_bitset& flip(unsigned int pos) ;
    
    
    inline bool operator[](unsigned int pos)
      const 
    {
#ifndef FAST_CODE
      if(pos >= seqlen)
	throw std::out_of_range("index bigger than bitset length");
#endif
      
      unsigned int rpos = pos % (sizeof(unsigned long)*8);
      pos /= (sizeof(unsigned long)*8);
      
      return ((bool)((this->memory[pos] >> rpos) & 1));
    }
    
    inline std::string to_string() const;
    inline std::string to_hexstring() const;
    inline whiteice::math::integer to_integer() const;
    
    inline unsigned int count() const ; // number of bits set
    inline unsigned int size() const ;  // length of the bit sequence
        
    inline unsigned int blocks() const ; // number of 8bit blocks
    
    inline void resize(unsigned int len) ;  // sets length of sequence
    
    inline bool operator==(const dynamic_bitset& r) const ;
    inline bool operator!=(const dynamic_bitset& r) const ;
    
    inline bool any() const ;  // is any bit set?
    inline bool none() const ; // no bits are set?
    
    inline dynamic_bitset operator~() const; // returns flipped bitset
    inline bool operator!() const; // returns true if no bits are set (iszero)
    inline bool operator<(const dynamic_bitset& rh) const;
    
    // shifts bitset cyclically (pos > 0: left shift, pos < 0: right shift)
    inline void cyclicshift(int pos) ;
    
    // accesses 8 bit blocks
    // WARNING: when accessing the last block's unused bits
    // must be set to zero or dynamic_bitset may stop working correctly
    inline unsigned char& value(unsigned int p) {
#ifndef FAST_CODE
      // remove checks to increase speed and reduce stability
      if(p >= (sizeof(unsigned long)*numblocks)/sizeof(unsigned char))
	throw std::out_of_range("index bigger than bitset length");
#endif
    
      return ((unsigned char*)(this->memory))[p];
    }
    
    inline const unsigned char& value(unsigned int p) const {
#ifndef FAST_CODE
      // remove checks to increase speed and reduce stability
      if(p >= (sizeof(unsigned long)*numblocks)/sizeof(unsigned char))
	throw std::out_of_range("index bigger than bitset length");
#endif
      
      return ((unsigned char*)memory)[p];
    }
    
    
    // increments unsigned integer by one 
    // (uses carry flag to optimize this one)
    inline void inc() 
    {      
#if defined IA32
      // assumes sizeof(unsigned long) = 4
      
      asm("1:             \n\t" // loop start
	  "incl (%%eax)   \n\t"
	  "jnz 2f         \n\t"
	  "addl $4, %%eax \n\t"
	  "decl %%ecx     \n\t"
	  "jnz 1b         \n\t"
	  "2:             \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#elif defined AMD64
      // assumes sizeof(unsigned long) = 8
      
      asm("1:             \n\t" // loop start
	  "incq (%%rax)   \n\t"
	  "jnz 2f         \n\t"
	  "addq $8, %%rax \n\t"
	  "decl %%ecx     \n\t"
	  "jnz 1b         \n\t"
	  "2:             \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#else
#error "Unsupported machine architecture"
#endif
      
      clean_dirty_bits();
    }
    
    
    // decrements unsigned integer by one
    inline void dec() 
    {

#if defined IA32
      // assumes sizeof(unsigned long) = 4
      
      asm("1:               \n\t" // loop start
	  "subl $1, (%%eax) \n\t" // no way to get this done with dec?
	  "jnc 2f           \n\t" // (dec doesn't update carry flag)
	  "addl $4, %%eax   \n\t"
	  "decl %%ecx       \n\t"
	  "jnz 1b           \n\t"
	  "2:               \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#elif defined AMD64
      // assumes sizeof(unsigned long) = 8
      
      asm("1:               \n\t" // loop start
	  "subq $1, (%%rax) \n\t" // no way to get this done with dec?
	  "jnc 2f           \n\t" // (dec doesn't update carry flag)
	  "addq $8, %%rax   \n\t"
	  "decl %%ecx       \n\t"
	  "jnz 1b           \n\t"
	  "2:               \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));      
#else
#error "Unsupported machine architecture"
#endif
      
      clean_dirty_bits();
    }
    


    
  private:
    
    inline void clean_dirty_bits(){ // cleans unused bits
      // all zero mask isn't possible/it codes: no cleaning needed case
      if(cleanmask) 
	memory[numblocks - 1] &= cleanmask;
    }
    
    // needs to be done every time seqlen changes
    inline void calculate_cleanmask();
    
    
    unsigned long* memory; // nbits worth memory (+ round up to sizeof(char)*8 bits)
    
    unsigned int seqlen; // lengts of bit sequence (in bits)
    unsigned int numblocks; // lenght of sequence (in unsigned longs)
    
    unsigned long cleanmask; // used to clean possible dirty bits
    
  };
  
  
  
  
  inline std::ostream& operator<<(std::ostream& os, const dynamic_bitset& b);
  
  
};



/*
 * calculate_cleanmask() must be called
 * every time seqlen changes (so that cleanmask
 * gets updated and clean_dirty_bits() works correctly)
 *
 */

//#include "dynamic_bitset.h"

#include <new>
#include <stdexcept>
#include <exception>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

namespace whiteice
{


  inline dynamic_bitset::dynamic_bitset()
  {
    memory = 0;        
    seqlen = 0;
    numblocks = 0;
    
    calculate_cleanmask();
  }
  
  

  inline dynamic_bitset::dynamic_bitset(const dynamic_bitset& bitset)
  {
    memory = 0;        
    seqlen = 0;
    numblocks = 0;
    cleanmask = 0;
    
    if(bitset.seqlen > 0){
      
      numblocks = bitset.numblocks;
      
      memory = (unsigned long*)malloc(numblocks*sizeof(unsigned long));
      if(memory == 0) throw std::bad_alloc();
      
      seqlen = bitset.seqlen;
      cleanmask = bitset.cleanmask;
      
      memcpy(memory, bitset.memory, numblocks*sizeof(unsigned long));
    }
  }
  

  inline dynamic_bitset::dynamic_bitset(unsigned long val)
  {
    memory = 0;        
    seqlen = 0;
    numblocks = 0;        
    
    unsigned int blocksize = sizeof(unsigned long)*8; // in bits      
    numblocks = (8*sizeof(unsigned long) + blocksize - 1)/ blocksize;
    
    memory = (unsigned long*)malloc(numblocks*sizeof(unsigned long));
    if(!memory) throw std::bad_alloc();
    
    seqlen = 8*sizeof(unsigned long);
    calculate_cleanmask();
    
    memcpy(memory, &val, numblocks*sizeof(unsigned long));
    
    clean_dirty_bits();
  }

  
  inline dynamic_bitset::dynamic_bitset(const whiteice::math::integer& i)
  {
    const unsigned long int B = i.bits();
    
    memory = 0;    
    seqlen = 0;
    numblocks = 0;        
    
    unsigned int blocksize = sizeof(unsigned long)*8; // in bits 
    numblocks = (B + blocksize - 1)/ blocksize;
    
    memory = (unsigned long*)malloc(numblocks*sizeof(unsigned long));
    if(!memory) throw std::bad_alloc();
    
    seqlen = B;
    calculate_cleanmask();
    
    // is there faster way to do this?
    // 
    for(unsigned long int b=0;b<B;b++)
      set(b, i.getbit(b));
    
    
    clean_dirty_bits();
  }
  

  inline dynamic_bitset::~dynamic_bitset(){
    if(memory) free(memory);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::operator=(const dynamic_bitset& bitset)
  {    
    
    if(bitset.seqlen > 0){
      
      unsigned long* ptr = (unsigned long*)realloc(memory, bitset.numblocks*sizeof(unsigned long));
      if(!ptr) throw std::bad_alloc();
      
      memory = ptr;
      
      numblocks = bitset.numblocks;
      seqlen = bitset.seqlen;
      cleanmask = bitset.cleanmask;
      
      memcpy(memory, bitset.memory, numblocks*sizeof(unsigned long));
    }
    else{
      
      if(memory) free(memory);
      
      memory = 0;        
      seqlen = 0;
      numblocks = 0;
      cleanmask = 0;
    }
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::operator&=(const dynamic_bitset& r)
  {
    if(this->seqlen != r.seqlen)
      throw std::invalid_argument("Cannot and bitsets of different length");
    
    if(!seqlen) return (*this);
    
    for(unsigned int i=0;i<numblocks;i++){
      memory[i] &= r.memory[i];
    }
    
    
    clean_dirty_bits();
    
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::operator|=(const dynamic_bitset& r)
  {
    if(this->seqlen != r.seqlen)
      throw std::invalid_argument("Cannot and bitsets of different length");
    
    if(!seqlen) return (*this);
    
    for(unsigned int i=0;i<numblocks;i++){
      memory[i] |= r.memory[i];
    }
    
    
    clean_dirty_bits();
    
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::operator^=(const dynamic_bitset& r)
  {
    if(this->seqlen != r.seqlen)
      throw std::invalid_argument("Cannot and bitsets of different length");
    
    if(!seqlen) return (*this);
    
    for(unsigned int i=0;i<numblocks;i++){
      memory[i] ^= r.memory[i];
    }
    
    clean_dirty_bits();
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::operator<<=(int pos) 
  {
    if(pos < 0){
      return ((*this) >>= -pos);
    }
    else if(pos == 0){
      return ((*this));
    }
    else if(((unsigned)pos) >= seqlen){
      reset();
      return (*this);
    }
    
    const unsigned int blocksize = sizeof(unsigned long)*8;
    
    unsigned int cpos = pos % blocksize;
    pos /= blocksize;
    
    if(pos > 0){
      memmove(&(memory[pos]), memory, (numblocks - pos)*sizeof(unsigned long));
      memset(memory, 0x00, pos*sizeof(unsigned long));
    }
    
    if(cpos == 0 ){
      clean_dirty_bits();
      return (*this);
    }
    
    // shift to left from memory[pos] by cpos bits
    
    unsigned long temp, shiftin;
    shiftin = 0;
    
    for(unsigned int i=pos;i<numblocks;i++){
      temp = memory[i] >> (blocksize - cpos); // part which moves to the next block
      memory[i] <<= cpos;
      memory[i] |= shiftin;
      shiftin = temp;
    }
    
    
    // clears non used part of the final long
    clean_dirty_bits();
    
    return (*this);
  }
  
  
  

  inline dynamic_bitset& dynamic_bitset::operator>>=(int pos) 
  {
    if(pos < 0){
      return ((*this) <<= -pos);
    }
    else if(pos == 0){
      return ((*this));
    }
    else if(((unsigned)pos) >= seqlen){
      reset();
      return (*this);
    } 
    
    const unsigned int blocksize = sizeof(unsigned long)*8;
    
    unsigned int cpos = pos % blocksize;
    pos /= blocksize;
    
    memmove(memory, &(memory[pos]), (numblocks - pos)*sizeof(unsigned long));
    memset(&(memory[numblocks - pos]), 0x00, pos*sizeof(unsigned long));
    
    if(cpos == 0 ){
      clean_dirty_bits();
      return (*this);
    }
    
    // shift to right from memory[numblocks - pos - 1] by cpos bits
    
    // clears non used part of the final long
    // (so shift brings only zeros in the actual bitset)
    clean_dirty_bits();
    
    unsigned long temp, shiftin;
    shiftin = 0;
    
    unsigned int i = numblocks;
    
    do{
      i--;
      
      temp = memory[i] << (blocksize - cpos);
      
      memory[i] >>= cpos;
      memory[i] |= shiftin;
      shiftin = temp;
    } 
    while(i > 0);
    

    return (*this);
  }
  
  
  inline dynamic_bitset& dynamic_bitset::operator+=(const dynamic_bitset& r)
  {
    this->resize(this->size() + r.size());
    
    (*this) <<= r.size(); // note: does extra work: zeroes fullblock lowest part of this
    
    unsigned int nblocks = r.size() / (sizeof(unsigned long)*8);
    unsigned int nbits   = r.size() % (sizeof(unsigned long)*8);
    
    memcpy(memory, r.memory, nblocks*sizeof(unsigned long));
    
    // ORs the final parts of the bits in the place
    // (low part of this and high part of r are zero)
    
    if(nbits)
      memory[nblocks] |= r.memory[nblocks];
    
    return (*this);
  }
  
  
  inline dynamic_bitset dynamic_bitset::operator&(const dynamic_bitset& r) const
  {
    dynamic_bitset copy(*this);
    
    copy &= r;
    
    return copy;
  }
  
  

  inline dynamic_bitset dynamic_bitset::operator|(const dynamic_bitset& r) const
  {
    dynamic_bitset copy(*this);
    
    copy |= r;
    
    return copy;
  }
  
  

  inline dynamic_bitset dynamic_bitset::operator^(const dynamic_bitset& r) const
  {
    dynamic_bitset copy(*this);
    
    copy ^= r;
    
    return copy;
  }
  
  

  inline dynamic_bitset dynamic_bitset::operator<<(const int pos) const 
  {
    dynamic_bitset copy(*this);
    
    copy <<= pos;
    
    return copy;
  }
  
  
  inline dynamic_bitset dynamic_bitset::operator+(const dynamic_bitset& r) const
  {
    dynamic_bitset copy(*this);
    
    copy += r;
    
    return copy;
  }
  
  
  inline dynamic_bitset dynamic_bitset::operator>>(const int pos) const 
  {
    dynamic_bitset copy(*this);
    
    copy >>= pos;
    
    return copy;
  }
  
  inline bool dynamic_bitset::operator<(const dynamic_bitset& rh) const
  {
    whiteice::math::integer i, j;

    i = this->to_integer();
    j = rh.to_integer();

    return (i < j);
  }
  

  inline dynamic_bitset& dynamic_bitset::set()   // sets all bits
  {
    memset(memory, 0xff, numblocks*sizeof(unsigned long));
    
    clean_dirty_bits();
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::set(unsigned int pos, bool val) 
  {
    if(!val) return reset(pos);
    
    if(pos >= seqlen)
      throw std::out_of_range("index bigger than bitset length");        
        
    unsigned int cpos = pos % (sizeof(unsigned long)*8);
    pos /= sizeof(unsigned long)*8;

    unsigned long setter = 1UL << cpos;
    
    memory[pos] |= setter;
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::reset()   // clears all bits
  {
    memset(memory, 0x00, numblocks*sizeof(unsigned long));
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::reset(unsigned int pos) 
  {
    if(pos >= seqlen)
      throw std::out_of_range("index bigger than bitset length");
    
    unsigned int cpos = pos % (sizeof(unsigned long)*8);
    pos /= (sizeof(unsigned long)*8);
    
    unsigned long xorer = (unsigned long)(-1L);
    
    unsigned long clearer = 1UL << cpos;
    clearer ^= xorer;
    
    memory[pos] &= clearer;
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::flip() 
  {
    if(!seqlen) return (*this);
    
    unsigned long xorer = (unsigned long)(-1L);
    
    for(unsigned int i=0;i<numblocks;i++)
      memory[i] ^= xorer;

    clean_dirty_bits();
    
    return (*this);
  }
  
  

  inline dynamic_bitset& dynamic_bitset::flip(unsigned int pos) 
  {
    if(pos >= seqlen)
      throw std::out_of_range("index bigger than bitset length");
    
    unsigned int cpos = pos % (sizeof(unsigned long)*8);
    pos /= sizeof(unsigned long)*8;
    
    unsigned long xorflip = 1UL << cpos;
    
    memory[pos] ^= xorflip;

    
    return (*this);
  }
  
  
  //  
  //
  //  integer<> dynamic_bitset::to_integer() const
  //  {
  //    integer<> i = 0;
  //    integer<> scale = 1;
  //    integer<> base = 1;
  //    
  //    // exponentation
  //    {
  //      unsigned int b = PRIMBITS;
  //      
  //      while(b > 0){
  //	if(base & 1){
  //	  base *= 2;
  //	  b--;
  //	}
  //	else{
  //	  base *= base; // square it
  //	  b >>= 1; // divide by two
  //	}
  //      }
  //    }
  //
  //  
  //  
  //     typename std::vector<std::bitset<PRIMBITS> >::const_iterator j =
  //      bits.begin();
  //    
  //    while(j != bits.end()){
  //      
  //      i += integer<>(j->to_long()) * scale;
  //      
  //      scale *= base;
  //      
  //      i++;
  //    }
  //    
  //    return i;
  //  }
  
  

  inline std::string dynamic_bitset::to_string() const
  {
    std::string str;
    
    unsigned int lbits;
    unsigned int i = numblocks;
    
    if(seqlen == 0) // empty bitset
      return std::string("[empty]");
    
    lbits = seqlen % (sizeof(unsigned long)*8);
    
    // processes first (possible) partial block
    if(lbits){
      
      i--;
      
      do{
	lbits--;
	
	if((memory[i] >> lbits) & 1UL) str += "1";
	else str += "0";
      }
      while(lbits > 0);
      
      if(i == 0) return str;
    }
    

    do{
      i--;
      
      lbits = sizeof(unsigned long)*8;
      
      do{
	lbits--;
	
	if((memory[i] >> lbits) & 1UL) str += "1";
	else str += "0";
      }
      while(lbits > 0);
      
    }
    while(i > 0);
    
    return str;
  }
  
  
  
  inline std::string dynamic_bitset::to_hexstring() const
  {
    std::string str;
    char buf[128];
    
    
    // prints in reversed order, highest block last
    unsigned int numcharblocks = 
      ((numblocks*sizeof(unsigned long))/sizeof(unsigned char));
    
    for(unsigned int i=0;i<numcharblocks;i++){
      sprintf(buf,"%02x", ((unsigned char*)memory)[numcharblocks - 1 - i]);
      str += buf;
    }
  
#if 0
    // broken???
    
    int i = numblocks;
    
    while(i > 0){
      i--;
      
      sprintf(buf,"%08lx", (unsigned long)memory[i]); // 8 = sizeof(long)*2
      
      str += buf;
    }
#endif
    
    return str;
  }
  
  
  
  inline whiteice::math::integer dynamic_bitset::to_integer() const
  {
    whiteice::math::integer cint;
    
    unsigned int lbits;
    unsigned int i = numblocks;
    unsigned int bitpos = seqlen;
    
    lbits = seqlen % (sizeof(unsigned long)*8);
    
    // processes first (possible) partial block
    if(lbits){
      
      i--;
      
      do{
	lbits--;
	bitpos--;
	
	if((memory[i] >> lbits) & 1UL)
	  cint.setbit(bitpos);
      }
      while(lbits > 0);
      
      if(i == 0) return cint;
    }
    

    do{
      i--;
      
      lbits = sizeof(unsigned long)*8;
      
      do{
	lbits--;
	bitpos--;
	
	if((memory[i] >> lbits) & 1UL)
	  cint.setbit(bitpos);
      }
      while(lbits > 0);
      
    }
    while(i > 0);
    
    return cint;
  }
  
  
  inline unsigned int dynamic_bitset::count() const  // number of bits set
  {    
    if(seqlen == 0) return 0;
    unsigned int c = 0;
    
    int lbits;        
    
    for(unsigned int i=0;i<numblocks;i++){ // longs
      
      lbits = sizeof(unsigned long)*8;
      
      do{
	lbits--;
	
	if((memory[i] >> lbits) & 1UL) c++;
      }
      while(lbits > 0);
      
    }
    
    return c;
  }
  
  

  inline unsigned int dynamic_bitset::size() const    // length of the bit sequence
  {
    return seqlen;
  }
  
  
  inline unsigned int dynamic_bitset::blocks() const 
  {
    return (numblocks*(sizeof(unsigned long)/sizeof(unsigned char)));
  }
  

  inline void dynamic_bitset::resize(unsigned int len)   // sets length of sequence
  {
    if(len == 0){
      
      if(memory) free(memory);
      
      memory = 0;
      seqlen = 0;
      numblocks = 0;
      cleanmask = 0;
    }
    
    
    unsigned int blocksize = sizeof(unsigned long)*8; // in bits
    unsigned int nb = (len + blocksize - 1)/ blocksize;
      
    unsigned long* ptr = 
      (unsigned long*)realloc(memory,nb*sizeof(unsigned long));
    
    if(ptr == 0) throw std::bad_alloc();
    
    memory = ptr;
    
    // resets new memory
    if(nb > numblocks)
      memset(&(memory[numblocks]), 0x00,
	     (nb - numblocks)*sizeof(unsigned long));        
    
    
    numblocks = nb;
    seqlen = len;

    calculate_cleanmask();    
    clean_dirty_bits();
  }
  
  

  inline bool dynamic_bitset::operator==(const dynamic_bitset& r) const 
  {
    if(r.seqlen != seqlen) return false;
        
    for(unsigned int i=0;i<numblocks;i++){
      if(memory[i] != r.memory[i])
	return false;
    }
    
    
    return true;
  }
  
  

  inline bool dynamic_bitset::operator!=(const dynamic_bitset& r) const 
  {
    return (!( (*this) == r ));
  }
  
  

  inline bool dynamic_bitset::any() const    // is any bit set?
  {
    
    for(unsigned int i=0;i<numblocks;i++){
      if(memory[i] != 0) return true;
    }
    
    return false;
  }
  
  

  inline bool dynamic_bitset::none() const   // no bits are set?
  {
    for(unsigned int i=0;i<numblocks;i++){
      if(memory[i] != 0) return false;
    }
    
    return true;
  }
  
  
  
  inline dynamic_bitset dynamic_bitset::operator~() const  // returns flipped bitset
  {
    dynamic_bitset b(*this);    
    b.flip();
    
    return b;
  }

  
  // returns true if no bits are set
  inline bool dynamic_bitset::operator!() const
  {
    return (count() == 0);
  }
  
  
  // needs to be done every time seqlen changes
  inline void dynamic_bitset::calculate_cleanmask()
  {
    if(seqlen == 0){
      cleanmask = 0;
      return;
    }
    
    const unsigned int blocksize = sizeof(unsigned long)*8;
      
    // clears non used part of the final long
    if(seqlen % blocksize){
      unsigned int c = seqlen % blocksize;
      
      if(c == 0){
	cleanmask = 0;
	return;
      }
      
      cleanmask = ((1UL << c) - 1UL);
    }
    else cleanmask = 0;
  }
  
  
  // shifts bitset cyclically (pos > 0: left shift, pos < 0: right shift)
  inline void dynamic_bitset::cyclicshift(int pos) 
  {
    // note: not very fast implementation
    
    if(pos > 0){ // left shift
      
      dynamic_bitset temp(*this);
      
      temp >>= (temp.size() - pos);      
      
      temp.resize(pos);
      this->resize(this->size() - pos);
      
      (*this) += temp;
      
    }
    else if(pos < 0){ // right shift;
      pos = -pos;
      
      dynamic_bitset temp(*this);
      
      temp >>= pos;      
      (*this) <<= (this->size() - pos);
      this->resize(pos);
      temp.resize(temp.size() - pos);
      
      (*this) += temp;
      
    }
    
  }
  
  
  
  inline std::ostream& operator<<(std::ostream& os, const dynamic_bitset& b)
  {
    // limited support for iostream flags()
    
    if(os.flags() & std::ios::hex){
      if(os.flags() & std::ios::showbase)
	os << "0x";
      
      os << b.to_hexstring();
    }
    else{
      os << b.to_string();
    }
    
    return os;
  }
  
};



#endif
