//
//  kokoro-tts-lib
//
import Foundation

// MARK: - Supporting Types

/// A struct that holds a punctuation mark along with its
/// line index and a one‐character code describing its position:
///   - "B": beginning,
///   - "E": end,
///   - "I": internal (in the middle),
///   - "A": alone.
struct MarkIndex {
    let index: Int
    let mark: String
    let position: String
}

/// A simple separator type (for example, a word separator).
struct Separator {
    let word: String
}

// MARK: - Punctuation Class

/// The Punctuation class “hides” punctuation during processing
/// and later restores it. (Backends may handle punctuation differently.)
class Punctuation {
    static let defaultMarks: String = ";:,.!?¡¿—…\"«»“”(){}[]"
    
    /// If punctuation is initialized from a string, we store the (unique)
    /// characters here. (If initialized from a regex, this remains nil.)
    private var _marks: String?
    
    /// The compiled regular expression used to find punctuation.
    private var _marksRe: NSRegularExpression
    
    /// A computed property to access the punctuation marks as a string.
    /// (If the punctuation was created from a regex, accessing this will cause a runtime error.)
    var marks: String {
        get {
            if let m = _marks {
                return m
            } else {
                fatalError("Punctuation was initialized from a regex; cannot access marks as a string.")
            }
        }
        set {
            // Create a string containing only the unique characters.
            let uniqueChars = Set(newValue)
            let uniqueMarks = String(uniqueChars)
            _marks = uniqueMarks
            let escapedMarks = NSRegularExpression.escapedPattern(for: uniqueMarks)
            let pattern = "(\\s*[" + escapedMarks + "]+\\s*)+"
            do {
                _marksRe = try NSRegularExpression(pattern: pattern, options: [])
            } catch {
                fatalError("Invalid regex pattern: \(pattern)")
            }
        }
    }
    
    private static func buildPunctuationPattern(from marks: String) -> String {
        // Ensure each character is escaped as needed for a character class.
        let escapedMarks = marks.map { char -> String in
            // Escape square brackets manually; for other characters we can use the built-in escaping.
            if char == "[" || char == "]" {
                return "\\\(char)"
            } else {
                return NSRegularExpression.escapedPattern(for: String(char))
            }
        }.joined()
        
        // This pattern will match one or more groups of:
        // optional whitespace, one or more punctuation characters, optional whitespace.
        return "(\\s*[" + escapedMarks + "]+\\s*)+"
    }
    
    // MARK: Initializers
    
    /// Initializes the punctuation processor using a string of marks.
    /// The default is Punctuation.defaultMarks.
    init(marks: String = Punctuation.defaultMarks) {
        let uniqueChars = Set(marks)
        let uniqueMarks = String(uniqueChars)
        self._marks = uniqueMarks
        let pattern = Punctuation.buildPunctuationPattern(from: self._marks!)
        do {
            self._marksRe = try NSRegularExpression(pattern: pattern, options: [])
        } catch {
            fatalError("Invalid regex pattern: \(pattern)")
        }
    }
    
    // MARK: - Removal Method
    
    /// Returns a new string where all punctuation (as defined by the regex)
    /// has been replaced by a single space and trimmed.
    func remove(text: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let result = _marksRe.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: " ")
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    /// Overloaded: Process an array of strings.
    func remove(texts: [String]) -> [String] {
        return texts.map { remove(text: $0) }
    }
    
    // MARK: - Preservation Methods
    
    /// “Preserves” punctuation in a single string.
    /// Returns a tuple: an array of string chunks and an array of MarkIndex.
    func preserve(text: String) -> ([String], [MarkIndex]) {
        return preserve(texts: [text])
    }
    
    /// “Preserves” punctuation in an array of strings.
    /// For each input line, splits it into chunks (removing punctuation)
    /// and stores the punctuation marks (with their positions) for later restoration.
    func preserve(texts: [String]) -> ([String], [MarkIndex]) {
        var preservedText: [String] = []
        var preservedMarks: [MarkIndex] = []
        for (num, line) in texts.enumerated() {
            let (lineParts, marks) = preserveLine(line: line, num: num)
            preservedText.append(contentsOf: lineParts)
            preservedMarks.append(contentsOf: marks)
        }
        // Filter out any empty strings.
        let filteredText = preservedText.filter { !$0.isEmpty }
        return (filteredText, preservedMarks)
    }
    
    /// Internal helper: processes a single line.
    /// – Returns: A tuple where the first element is an array of text chunks
    ///   (the parts of the line between punctuation) and the second element is
    ///   an array of MarkIndex describing the punctuation that was removed.
    private func preserveLine(line: String, num: Int) -> ([String], [MarkIndex]) {
        let nsLine = line as NSString
        let range = NSRange(location: 0, length: nsLine.length)
        let matches = _marksRe.matches(in: line, options: [], range: range)
        if matches.isEmpty {
            return ([line], [])
        }
        // If the entire line is punctuation...
        if matches.count == 1, let match = matches.first, match.range.length == nsLine.length {
            let markIndex = MarkIndex(index: num, mark: line, position: "A")
            return ([], [markIndex])
        }
        
        // Build the list of punctuation marks with their positions.
        var marks: [MarkIndex] = []
        for (i, match) in matches.enumerated() {
            let matchStr = nsLine.substring(with: match.range)
            var position = "I"
            if i == 0, line.hasPrefix(matchStr) {
                position = "B"
            } else if i == matches.count - 1, line.hasSuffix(matchStr) {
                position = "E"
            }
            marks.append(MarkIndex(index: num, mark: matchStr, position: position))
        }
        
        // Split the line into chunks. The Python version repeatedly splits the line
        // at the first occurrence of each punctuation mark.
        var preservedLine: [String] = []
        var remainingLine = line
        for mark in marks {
            if let markRange = remainingLine.range(of: mark.mark) {
                let prefix = String(remainingLine[..<markRange.lowerBound])
                let suffix = String(remainingLine[markRange.upperBound...])
                preservedLine.append(prefix)
                remainingLine = suffix
            }
        }
        preservedLine.append(remainingLine)
        return (preservedLine, marks)
    }
    
    // MARK: - Restoration Method
    
    /// Restores punctuation into a processed text.
    ///
    /// - Parameters:
    ///   - text: An array of text chunks (for example, as returned by `preserve`).
    ///   - marks: The list of punctuation marks (with positions) previously removed.
    ///   - sep: A separator (whose `word` is inserted if needed).
    ///   - strip: If false, the word separator is appended when necessary.
    /// - Returns: An array of strings with punctuation restored.
    static func restore(text: [String], marks: [MarkIndex], sep: Separator, strip: Bool) -> [String] {
        var texts = text
        var marks = marks
        var punctuatedText: [String] = []
        var pos = 0
        
        while !texts.isEmpty || !marks.isEmpty {
            if marks.isEmpty {
                // If no punctuation marks remain, output what is left.
                for var line in texts {
                    if !strip, !sep.word.isEmpty, !line.hasSuffix(sep.word) {
                        line.append(sep.word)
                    }
                    punctuatedText.append(line)
                }
                texts.removeAll()
            } else if texts.isEmpty {
                // Nothing has been “phonemized”; return the punctuation marks joined together,
                // with internal spaces replaced by the separator.
                let combinedMark = marks.map { $0.mark }.joined()
                let replaced = combinedMark.replacingOccurrences(of: " ", with: sep.word)
                punctuatedText.append(replaced)
                marks.removeAll()
            } else {
                let currentMark = marks.first!
                if currentMark.index == pos {
                    let mark = currentMark.mark.replacingOccurrences(of: " ", with: sep.word)
                    marks.removeFirst()
                    // Remove a trailing separator from the current text, if present.
                    if !sep.word.isEmpty, texts[0].hasSuffix(sep.word) {
                        texts[0] = String(texts[0].dropLast(sep.word.count))
                    }
                    switch currentMark.position {
                    case "B":
                        texts[0] = mark + texts[0]
                    case "E":
                        var line = texts[0] + mark
                        if !strip, !sep.word.isEmpty, !mark.hasSuffix(sep.word) {
                            line.append(sep.word)
                        }
                        punctuatedText.append(line)
                        texts.removeFirst()
                        pos += 1
                    case "A":
                        var line = mark
                        if !strip, !sep.word.isEmpty, !mark.hasSuffix(sep.word) {
                            line.append(sep.word)
                        }
                        punctuatedText.append(line)
                        pos += 1
                    default: // "I"
                        if texts.count == 1 {
                            texts[0] = texts[0] + mark
                        } else {
                            let firstWord = texts[0]
                            texts.removeFirst()
                            texts[0] = firstWord + mark + texts[0]
                        }
                    }
                } else {
                    punctuatedText.append(texts[0])
                    texts.removeFirst()
                    pos += 1
                }
            }
        }
        return punctuatedText
    }
}

