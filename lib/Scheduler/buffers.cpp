#include "opt-sched/Scheduler/buffers.h"
#include "opt-sched/Scheduler/logger.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

using namespace llvm::opt_sched;

#ifdef WIN32
#include <io.h>

static const int INPFILE_OPENFLAGS = _O_BINARY | _O_RDONLY;
#else
#include <sys/uio.h>

static const int INPFILE_OPENFLAGS = O_RDONLY;
#endif

static const char CR = '\r';
static const char LF = '\n';
static const char SPC = ' ';
static const char TAB = '\t';
static const char CMNT_STRT = '#';
static const int EOL = -2;
static const int DATA = 0;
static const int FILEOPEN_ERROR = -1;

static inline bool IsWhitespace(const char &c) { return c == SPC || c == TAB; }
static inline bool IsLineEnd(const char &c) { return c == CR || c == LF; }
static inline bool IsCommentStart(const char &c) { return c == CMNT_STRT; }

InputBuffer::InputBuffer() {
  fileHndl = FILEOPEN_ERROR;
  buf = NULL;
  totSize = DFLT_INPBUF_SIZE;
  crntLineNum = 0;
  crntLineOfst = 0;
}

InputBuffer::~InputBuffer() {
  Clean();
  Unload();
}

FUNC_RESULT InputBuffer::Load(const char *const fileName,
                              const char *const path, long maxByts) {
  char fullPath[MAX_NAMESIZE];
  strcpy(fullPath, path);
  strcat(fullPath, "\\");
  strcat(fullPath, fileName);
  return Load(fullPath, maxByts);
}

FUNC_RESULT InputBuffer::Load(const char *const _fullPath, long maxByts) {
  strcpy(fullPath, _fullPath);

  if ((fileHndl = open(fullPath, INPFILE_OPENFLAGS)) == FILEOPEN_ERROR) {
    perror("\nSystem Error in opt_sched ");
    Logger::Fatal("Error openning input file: %s.", fullPath);
    return RES_ERROR;
  }

  totSize = maxByts;

  if (totSize > 0x7fffffff) {
    Logger::Fatal("Too large input file buffer.");
    return RES_ERROR;
  }

  if (buf != NULL) {
    delete[] buf;
  }

  // Allocate an extra byte for possible null termination.
  buf = new char[totSize + 1];
  if ((loadedByts = read(fileHndl, buf, totSize)) == 0) {
    Logger::Fatal("Empty input file: %s.", fullPath);
  }

  crntOfst = 0;
  crntLineNum = 1;
  crntLineOfst = 0;
  lineEndOfst = -1;
  lastChnk = loadedByts < totSize ? true : false;
  cmnt = false;
  lineStrt = true;

  if (skipSpaceAndCmnts() == NXT_EOF) {
    Logger::Fatal("No actual data in input file: %s", fullPath);
  }

  return RES_SUCCESS;
}

FUNC_RESULT InputBuffer::SetBuf(char *_buf, long size) {
  strcpy(fullPath, "NONE");
  fileHndl = FILEOPEN_ERROR;
  totSize = size;

  if (totSize > 0x7fffffff) {
    Logger::Error("Too large input buffer.");
    return RES_ERROR;
  }

  buf = _buf;

  if (buf == NULL) {
    Logger::Error("Invalid input buffer.");
    return RES_ERROR;
  }

  loadedByts = totSize;

  if (loadedByts <= 0) {
    Logger::Error("Empty input buffer.");
    return RES_ERROR;
  }

  crntOfst = 0;
  crntLineNum = 1;
  crntLineOfst = 0;
  lineEndOfst = -1;
  lastChnk = true;
  cmnt = false;
  lineStrt = true;

  if (skipSpaceAndCmnts() == NXT_EOF) {
    Logger::Error("No actual data in input buffer.");
    return RES_ERROR;
  }

  return RES_SUCCESS;
}

int InputBuffer::Reload() {
  if (lastChnk) {
    return RES_SUCCESS;
  }

  assert(crntOfst == crntLineOfst);
  long bytsNeeded = loadedByts - crntOfst;

  // If some bytes from the current chunck are still needed.
  if (bytsNeeded > 0) {
    // Move these bytes to the beginning of the buffer.
    memmove(buf, buf + crntOfst, bytsNeeded);
  }

  if ((loadedByts = read(fileHndl, buf + bytsNeeded, totSize - bytsNeeded)) ==
      0) {
    return EOF;
  }

  loadedByts += bytsNeeded;
  lastChnk = (loadedByts < totSize) ? true : false;
  crntLineOfst = 0;
  crntOfst = 0;
  return RES_SUCCESS;
}

int InputBuffer::chckReload() {
  if (crntOfst < loadedByts) {
    return RES_SUCCESS;
  } else if (lastChnk) {
    return EOF;
  } else {
    return Reload();
  }
}

NXTLINE_TYPE InputBuffer::skipSpaceAndCmnts() {
  bool emptyLineEncountered = false;
  // Line that starts with blank.
  bool blankedLine = false;
  char crntChar;

  while (true) {
    crntChar = buf[crntOfst];

    if (lineStrt) {
      // If previous line started with blank.
      if (blankedLine)
        emptyLineEncountered = true;
      blankedLine = false;

      if (IsLineEnd(crntChar)) {
        emptyLineEncountered = true;
      } else if (IsWhitespace(crntChar)) {
        blankedLine = true;
      } else if (!IsCommentStart(crntChar)) {
        return emptyLineEncountered ? NXT_SPC : NXT_DATA;
      }
    } else if (blankedLine) {
      if (!IsWhitespace(crntChar) && !IsCommentStart(crntChar)) {
        return emptyLineEncountered ? NXT_SPC : NXT_DATA;
      }
    }

    if (++crntOfst >= loadedByts)
      if (chckReload() == EOF) {
        return NXT_EOF;
      }

    // If current char is LF then next is a line start.
    lineStrt = (crntChar == LF);
  }
}

int InputBuffer::skipSpace() {
  for (crntChar = buf[crntOfst]; IsWhitespace(crntChar);
       crntChar = buf[crntOfst]) {
    if (++crntOfst >= loadedByts && chckReload() == EOF)
      return EOF;
  }

  if (IsCommentStart(crntChar))
    return CMNT_STRT;

  if (IsLineEnd(crntChar)) {
    // If at end of line go to start of next line.
    int step = (crntChar == CR) ? 2 : 1;
    // Skip [CR] and LF.
    crntOfst += step;

    if (crntOfst >= loadedByts && chckReload() == EOF)
      return EOF;

    lineStrt = true;
    nxtLineRchd = true;
    return EOL;
  }

  return DATA;
}

int InputBuffer::skipCmnt() {
  for (crntChar = buf[crntOfst]; !IsLineEnd(crntChar);
       crntChar = buf[crntOfst]) {
    if (++crntOfst >= loadedByts && chckReload() == EOF)
      return EOF;
  }

  int step = (crntChar == CR) ? 2 : 1;
  // Skip LF.
  crntOfst += step;

  if (crntOfst >= loadedByts && chckReload() == EOF)
    return EOF;

  lineStrt = true;
  nxtLineRchd = true;
  return EOL;
}

NXTLINE_TYPE InputBuffer::GetNxtVldLine_(int &pieceCnt, char *strng[],
                                         int lngth[], int maxPieceCnt) {
  crntLineOfst = crntOfst;

  // If the next line might not fit entirely into the buffer.
  if ((totSize - crntOfst) < INBUF_MAX_LINESIZE && !lastChnk) {
    // Flush the buffer by reloading a new chunck into it.
    Reload();
  }

  for (pieceCnt = 0, nxtLineRchd = false; !nxtLineRchd;) {
    assert(pieceCnt < maxPieceCnt);
    lngth[pieceCnt] = 0;
    strng[pieceCnt] = buf + crntOfst;

    for (crntChar = buf[crntOfst]; !IsWhiteSpaceOrLineEnd(crntChar);
         crntChar = buf[crntOfst]) {
      // Assume comments are always preceded by space.
      lngth[pieceCnt]++;

      if (++crntOfst >= loadedByts && chckReload() == EOF) {
        pieceCnt++;
        return NXT_EOF;
      }
    }

    if (lngth[pieceCnt] > 0)
      pieceCnt++;

    // If crntChar is CR or LF this will just go to the next line.
    switch (skipSpace()) {
    case DATA:
      if (pieceCnt == maxPieceCnt)
        return NXT_ERR;
      break;
    case EOL:
      break;
    case CMNT_STRT:
      if (skipCmnt() == EOF)
        return NXT_EOF;
      break;
    case EOF:
      return NXT_EOF;
    }
  }

  // At this point we should be at the beginning of the next line, which could
  // be space or comment.
  return skipSpaceAndCmnts();
}

void InputBuffer::Clean() {
  if (buf) {
    delete[] buf;
    buf = NULL;
  }
}

void InputBuffer::Unload() {
  if (fileHndl != -1) {
    close(fileHndl);
    fileHndl = -1;
  }
}

NXTLINE_TYPE InputBuffer::GetNxtVldLine(int &pieceCnt, char *strngs[],
                                        int lngths[]) {
  NXTLINE_TYPE retVal = GetNxtVldLine_(pieceCnt, strngs, lngths);

  for (int i = 0; i < pieceCnt; i++) {
    (strngs[i])[lngths[i]] = 0;
  }

  return retVal;
}

bool InputBuffer::IsWhiteSpaceOrLineEnd(char ch) {
  if (IsWhitespace(ch)) {
    return true;
  } else if (IsLineEnd(ch)) {
    if (crntOfst != lineEndOfst) {
      crntLineNum++;
      lineEndOfst = crntOfst;
    }
    return true;
  } else {
    return false;
  }
}

void InputBuffer::ReportError(char *msg, char *lineStrt, int frstLngth) {
  Logger::Error("%s on line %d of input file: %.*s. Line starts with: %s", msg,
                crntLineNum, fullPath, frstLngth, lineStrt);
}

void InputBuffer::ReportFatalError(char *msg, char *lineStrt, int frstLngth) {
  Logger::Fatal("%s on line %d of input file: %.*s. Line starts with: %s", msg,
                crntLineNum, fullPath, frstLngth, lineStrt);
}

void SpecsBuffer::ReadSpec(const char *const title, char *value) {
  int lngth[INBUF_MAX_PIECES_PERLINE];
  char *strPtr[INBUF_MAX_PIECES_PERLINE];
  int pieceCnt;
  bool isMltplPieces = false;
  bool isMsng = false;
  int totLngth;

  if (nxtLineType == NXT_EOF) {
    Logger::Fatal("End of Specs file unexpectedly encountered.");
  }

  nxtLineType = GetNxtVldLine(pieceCnt, strPtr, lngth);

  if (pieceCnt == 1) {
    isMsng = true;
  }

  if (pieceCnt > 2) {
    if (memcmp(strPtr[1], "\"", 1) == 0) {
      isMltplPieces = true;
    } else {
      Logger::Fatal("Invalid # of words on a line in spec file.");
    }
  }

  // Check if title is correct.
  if (strncmp(strPtr[0], title, lngth[0]) != 0) {
    // null terminate the string.
    strPtr[0][lngth[0]] = 0;
    Logger::Fatal("Invalid or misplaced title (%s) in specs file. Expected %s.",
                  strPtr[0], title);
  }

  if (isMltplPieces) {
    CombinePieces_(lngth, strPtr, 1, pieceCnt - 1, value, totLngth);
  } else if (isMsng) {
    strcpy(value, "unknown");
  } else {
    memcpy(value, strPtr[1], lngth[1]);
    value[lngth[1]] = 0;
  }
}

void SpecsBuffer::CombinePieces_(int lngths[], char *strngs[], int startPiece,
                                 int endPiece, char *target, int &totLngth) {
  int ofst = 0;

  for (int i = startPiece; i <= endPiece; i++) {
    memcpy(target + ofst, strngs[i], lngths[i]);
    ofst += lngths[i];
    target[ofst] = ' ';
    ofst++;
  }

  target[ofst] = 0;
  totLngth = ofst;
}

void SpecsBuffer::readLstElmnt(char *value) {
  int lngth[INBUF_MAX_PIECES_PERLINE];
  char *strPtr[INBUF_MAX_PIECES_PERLINE];
  int pieceCnt;

  if (nxtLineType == NXT_EOF) {
    Logger::Fatal("Unexpectedly encountered end of file %s", fullPath);
  }

  nxtLineType = GetNxtVldLine(pieceCnt, strPtr, lngth);

  if (pieceCnt != 1) {
    Logger::Fatal("Invalid number of tockens in file %s", fullPath);
  }

  memcpy(value, strPtr[0], lngth[0]);
  value[lngth[0]] = 0;
}

int SpecsBuffer::readIntLstElmnt() {
  char strVal[INBUF_MAX_LINESIZE];
  readLstElmnt(strVal);
  return atoi(strVal);
}

void SpecsBuffer::readLine(char *value, int maxPieceCnt) {
  int i, lngth[INBUF_MAX_PIECES_PERLINE];
  char *strPtr[INBUF_MAX_PIECES_PERLINE];
  int pieceCnt, ofst;

  assert(maxPieceCnt <= INBUF_MAX_PIECES_PERLINE);

  if (nxtLineType == NXT_EOF) {
    Logger::Fatal("End of Specs file unexpectedly encountered.");
  }

  nxtLineType = GetNxtVldLine(pieceCnt, strPtr, lngth);

  for (i = 0, ofst = 0; i < pieceCnt; i++) {
    memcpy(value + ofst, strPtr[i], lngth[i]);
    ofst += lngth[i];
    *(value + ofst) = ' ';
    ofst++;
  }

  // Null terminate the concatenated string.
  value[ofst] = 0;

  if (nxtLineType == NXT_ERR) {
    Logger::Fatal("Too many pieces on line: %s of input file: %s", value,
                  GetFullPath());
  }
}

bool SpecsBuffer::ReadFlagSpec(const char *const title, bool dfltValue) {
  char tmpStrng[MAX_NAMESIZE];

  ReadSpec(title, tmpStrng);

  if (strcmp(tmpStrng, "YES") == 0) {
    return true;
  } else if (strcmp(tmpStrng, "NO") == 0) {
    return false;
  }

  Logger::Error("Invalid value for (%s) flag in specs file. Defaulted to %s.",
                title, dfltValue ? "YES" : "NO");
  return dfltValue;
}

unsigned long SpecsBuffer::ReadUlongSpec(const char *const title) {
  char tmpStrng[MAX_NAMESIZE];
  ReadSpec(title, tmpStrng);
  return strtoul(tmpStrng, NULL, 10);
}

float SpecsBuffer::ReadFloatSpec(const char *const title) {
  char tmpStrng[MAX_NAMESIZE];

  ReadSpec(title, tmpStrng);
  return (float)atof(tmpStrng);
}

uint64_t SpecsBuffer::readUInt64Spec(const char *const title) {
  char tmpStrng[MAX_NAMESIZE];
  // Most sig piece and least sig piece.
  unsigned long MSUlong, LSUlong;
  int16_t ofst = 0, MSUlongSize, LSOfst;
  uint64_t fullNum;

  ReadSpec(title, tmpStrng);
  size_t lngth = strlen(tmpStrng);

  if (memcmp(tmpStrng, "0x", 2) == 0) {
    ofst = 2;
    lngth -= 2;
  } else {
    Logger::Error("0x missing in field %s in spec file %s. Hexadecimal assumed",
                  title, fullPath);
  }

  MSUlongSize = (lngth <= 8) ? 0 : (int16_t)lngth - 8;
  LSOfst = ofst + MSUlongSize;
  LSUlong = strtoul(tmpStrng + LSOfst, NULL, 16);
  MSUlong = 0;

  if (MSUlongSize > 0) {
    // We have already read the LS part.
    tmpStrng[ofst + MSUlongSize] = 0;
    MSUlong = strtoul(tmpStrng + ofst, NULL, 16);
  }

  // If MSUlong is zero these two lines will have no effect.
  fullNum = MSUlong;
  fullNum <<= 32;

  fullNum += LSUlong;
  return fullNum;
}

int16_t SpecsBuffer::ReadShortSpec(const char *const title) {
  char tmpStrng[MAX_NAMESIZE];
  ReadSpec(title, tmpStrng);
  return atoi(tmpStrng);
}

int SpecsBuffer::ReadIntSpec(const char *const title) {
  char tmpStrng[MAX_NAMESIZE];
  ReadSpec(title, tmpStrng);
  return (int16_t)atoi(tmpStrng);
}

void SpecsBuffer::ErrorHandle(char *value) {
  Logger::Fatal("Invalid parameter or spec (%s) in specs file.", value);
}

FUNC_RESULT SpecsBuffer::checkTitle(const char *const title) {
  int lngth[INBUF_MAX_PIECES_PERLINE];
  char *strPtr[INBUF_MAX_PIECES_PERLINE];
  int pieceCnt;

  if (nxtLineType == NXT_EOF) {
    Logger::Error("Unexpectedly encountered end of file %s.", fullPath);
    return RES_ERROR;
  }

  nxtLineType = GetNxtVldLine(pieceCnt, strPtr, lngth);

  if (pieceCnt != 1) {
    Logger::Error("Invalid number of tockens in file %s. Expected %s.",
                  fullPath, title);
    return RES_ERROR;
  }

  // Check if title is correct.
  if (strncmp(strPtr[0], title, lngth[0]) != 0) {
    // Null terminate the string.
    strPtr[0][lngth[0]] = 0;
    Logger::Error("Invalid or misplaced title (%s) in file %s. Expected %s.",
                  strPtr[0], fullPath, title);
    return RES_ERROR;
  }

  return RES_SUCCESS;
}

SpecsBuffer::SpecsBuffer() { nxtLineType = NXT_DATA; }
