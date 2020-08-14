/*******************************************************************************
Description:  Defines input buffering classes that can be used for opening,
              loading, buffering and parsing input files using system-level I/O,
              which relies on the programmer to do his own buffering, unlike the
              standard I/O which handles the buffering and hides it from the
              programmer.
Author:       Ghassan Shobaki
Created:      Oct. 1997
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_BUFFERS_H
#define OPTSCHED_GENERIC_BUFFERS_H

#include "opt-sched/Scheduler/defines.h"

namespace llvm {
namespace opt_sched {

const int INBUF_MAX_PIECES_PERLINE = 30;
const int INBUF_MAX_LINESIZE = 10000;
const int DFLT_INPBUF_SIZE = 1000000;

// String buffer size limits for file/sample names.
const int MAX_NAMESIZE = 1000;

enum NXTLINE_TYPE { NXT_EOF, NXT_SPC, NXT_DATA, NXT_ERR };

// This is an input buffer class for loading, buffering and parsing an input
// file using system level I/O, where the application program is responsible for
// allocating an input buffer and loading the file into it in chunks the size of
// each chunk used in this class is determined by the DFLT_INPBUF_SIZE value
// defined above. The class provides methods for skipping white space and
// comments and reading one valid data line at a time.
// Lexing Assumptions:
//   - Files do not contain any invalid characters. So if a character is not a
//     control character (\r, \n, #, \t or space), it is a valid data character.
//   - Comments on data lines should be preceded by at least one space character
//   - All files are scanned linewise
class InputBuffer {
public:
  InputBuffer();
  ~InputBuffer();
  int Reload();
  void Clean();
  void Unload();
  char *GetBuf() { return buf; }
  const char *GetFullPath() const { return fullPath; }
  FUNC_RESULT Load(const char *const fileName, const char *const path,
                   long maxByts = DFLT_INPBUF_SIZE);
  FUNC_RESULT Load(const char *const fullPath, long maxByts = DFLT_INPBUF_SIZE);
  FUNC_RESULT SetBuf(char *buf, long size);

  // This function skips all comments and white spaces (tabs are not taken
  // into account), and does not return until it reaches a valid data line or
  // end of file. If at least one line starting with space is encountered on
  // the way, the return value will be NXT_SPC. It should always be called
  // when the current offset is at the first character of a line
  // (lineStrt==true).
  NXTLINE_TYPE skipSpaceAndCmnts();
  NXTLINE_TYPE GetNxtVldLine(int &pieceCnt, char *strngs[], int lngths[]);

protected:
  char *buf;

  long totSize,     // total size of the buffer
      loadedByts,   // number of bytes loaded
      crntOfst,     // current offset within the buffer
      lineEndOfst,  // the offset of the last LF or CR character seen
      crntLineOfst, // the offset of the current line
      crntLineNum;  // the current line number

  int fileHndl;
  char crntChar, prevChar;
  bool lastChnk, cmnt, lineStrt, nxtLineRchd;
  char fullPath[MAX_NAMESIZE];

  // Keeps going until it encounters a data character or a line start.
  int skipSpace();
  // Keeps going until it encounters a new line (assume no embedded comments).
  int skipCmnt();
  // Checks if reloading is necessary and does it or detects end of file.
  int chckReload();

  NXTLINE_TYPE GetNxtVldLine_(int &pieceCnt, char *str[], int lngth[],
                              int maxPieceCnt = INBUF_MAX_PIECES_PERLINE);
  bool IsWhiteSpaceOrLineEnd(char ch);
  void ReportError(char *msg, char *lineStrt, int frstLngth);
  void ReportFatalError(char *msg, char *lineStrt, int frstLngth);
};

// A specs buffer is an input buffer for parsing a typical input specification
// or configuration file whose format is line based, i.e., includes one spec
// or setting per line. This class includes one method for parsing one type
// of specs
class SpecsBuffer : public InputBuffer {
public:
  SpecsBuffer();
  void ReadSpec(const char *const title, char *value);
  void readLine(char *value, int maxPieceCnt);
  void readLstElmnt(char *value);
  int readIntLstElmnt();
  bool ReadFlagSpec(const char *const title, bool dfltValue);
  unsigned long ReadUlongSpec(const char *const title);
  float ReadFloatSpec(const char *const title);
  uint64_t readUInt64Spec(const char *const title);
  int ReadIntSpec(const char *const title);
  int16_t ReadShortSpec(const char *const title);
  FUNC_RESULT checkTitle(const char *const title);
  void ErrorHandle(char *value);

protected:
  NXTLINE_TYPE nxtLineType;
  void CombinePieces_(int lngths[], char *strngs[], int startPiece,
                      int endPiece, char *target, int &totLngth);
};

} // namespace opt_sched
} // namespace llvm

#endif
