//
//  kokoro-tts-lib
//

class TextNormalizer {
    private init() {}
    
    /// Translates a numeric/time string into a “spoken” form.
    private static func splitNum(_ match: String) -> String {
        // If it contains a dot, leave it unchanged.
        if match.contains(".") {
            return match
        } else if match.contains(":") {
            let parts = match.split(separator: ":").compactMap { Int($0) }
            if parts.count == 2 {
                let h = parts[0], m = parts[1]
                if m == 0 { return "\(h) o'clock" }
                else if m < 10 { return "\(h) oh \(m)" }
                else { return "\(h) \(m)" }
            }
            return match
        }
        // Otherwise assume a year-like number.
        guard let year = Int(match.prefix(4)) else { return match }
        if year < 1100 || year % 1000 < 10 { return match }
        let left = String(match.prefix(2))
        let rightStr = match.dropFirst(2).prefix(2)
        let right = Int(rightStr) ?? 0
        let s = match.hasSuffix("s") ? "s" : ""
        if (100...999).contains(year % 1000) {
            if right == 0 {
                return "\(left) hundred\(s)"
            } else if right < 10 {
                return "\(left) oh \(right)\(s)"
            }
        }
        return "\(left) \(right)\(s)"
    }

    /// Converts a money string (with a leading '$' or '£') to a spoken version.
    private static func flipMoney(_ match: String) -> String {
        let bill = match.first == "$" ? "dollar" : "pound"
        // If the last character is alphabetic, assume the unit is already given.
        if let last = match.last, last.isLetter {
            let amount = match.dropFirst()
            return "\(amount) \(bill)s"
        } else if !match.contains(".") {
            let amount = match.dropFirst()
            let s = (amount == "1") ? "" : "s"
            return "\(amount) \(bill)\(s)"
        } else {
            let trimmed = match.dropFirst()  // remove the currency symbol
            let parts = trimmed.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { return match }
            let b = String(parts[0])
            var cStr = String(parts[1])
            let s = (b == "1") ? "" : "s"
            if cStr.count < 2 { cStr = cStr.padding(toLength: 2, withPad: "0", startingAt: 0) }
            let c = Int(cStr) ?? 0
            let coins: String = {
                if match.first == "$" {
                    return (c == 1) ? "cent" : "cents"
                } else {
                    return (c == 1) ? "penny" : "pence"
                }
            }()
            return "\(b) \(bill)\(s) and \(c) \(coins)"
        }
    }

    /// Converts a decimal number string into a spoken version (digits after the point are separated by spaces).
    private static func pointNum(_ match: String) -> String {
        let parts = match.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else { return match }
        let a = String(parts[0])
        let b = parts[1].map { String($0) }.joined(separator: " ")
        return "\(a) point \(b)"
    }

    static func normalizeText(_ text: String) -> String {
        var result = text
        // 1. Replace special quotes with simple ones.
        result = result
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{2019}", with: "'")
        // 2. Replace guillemets with curly quotes then straight quotes.
        result = result
            .replacingOccurrences(of: "«", with: "\u{201C}")
            .replacingOccurrences(of: "»", with: "\u{201D}")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
        // 3. Replace parentheses with guillemets.
        result = result
            .replacingOccurrences(of: "(", with: "«")
            .replacingOccurrences(of: ")", with: "»")
        // 4. Replace various East‐Asian punctuation with standard punctuation (plus a trailing space).
        let eastAsian = "、。！，：；？"
        let standard  = ",.!,:;?"
        for (ea, st) in zip(eastAsian, standard) {
            result = result.replacingOccurrences(of: String(ea), with: "\(st) ")
        }
        
        // 5. Regex substitutions:
        result = result
            // Replace any whitespace character (other than space or newline) with a space.
            .replacingMatches(pattern: "[^\\S \\n]", with: " ")
            // Collapse multiple spaces.
            .replacingMatches(pattern: " {2,}", with: " ")
            // Remove spaces that occur only between newlines.
            .replacingMatches(pattern: "(?<=\\n) +(?=\\n)", with: "")
            // Abbreviation substitutions.
            .replacingMatches(pattern: "\\bD[Rr]\\.(?= [A-Z])", with: "Doctor")
            .replacingMatches(pattern: "\\b(?:Mr\\.|MR\\.(?= [A-Z]))", with: "Mister")
            .replacingMatches(pattern: "\\b(?:Ms\\.|MS\\.(?= [A-Z]))", with: "Miss")
            .replacingMatches(pattern: "\\b(?:Mrs\\.|MRS\\.(?= [A-Z]))", with: "Mrs")
            .replacingMatches(pattern: "\\betc\\.(?! [A-Z])", with: "etc")
            .replacingMatches(pattern: "(?i)\\b(y)eah?\\b", options: [.caseInsensitive], with: "$1e'a")
            // Use custom functions for numbers/times.
            .replacingMatches(pattern: "\\d*\\.\\d+|\\b\\d{4}s?\\b|(?<!:)\\b(?:[1-9]|1[0-2]):[0-5]\\d\\b(?!:)", using: { _, match in
                splitNum(match)
            })
            // Remove commas between digits.
            .replacingMatches(pattern: "(?<=\\d),(?=\\d)", with: "")
            // Process currency amounts.
            .replacingMatches(pattern: "(?i)[$£]\\d+(?:\\.\\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\\b|[$£]\\d+\\.\\d\\d?\\b", options: [.caseInsensitive], using: { _, match in
                flipMoney(match)
            })
            // Process decimal numbers (insert “point” and spaces).
            .replacingMatches(pattern: "\\d*\\.\\d+", using: { _, match in
                pointNum(match)
            })
            // Replace a hyphen between digits with " to ".
            .replacingMatches(pattern: "(?<=\\d)-(?=\\d)", with: " to ")
            // Insert a space before an S following a digit.
            .replacingMatches(pattern: "(?<=\\d)S", with: " S")
            // Replace an apostrophe-s after certain uppercase letters.
            .replacingMatches(pattern: "(?<=[BCDFGHJ-NP-TV-Z])'?s\\b", with: "'S")
            // After an X' replace S with lowercase s.
            .replacingMatches(pattern: "(?<=X')S\\b", with: "s")
            // Replace dots in sequences like initials with hyphens.
            .replacingMatches(pattern: "(?:[A-Za-z]\\.){2,} [a-z]", using: { _, match in
                match.replacingOccurrences(of: ".", with: "-")
            })
            // Replace a period between uppercase letters with a hyphen.
            .replacingMatches(pattern: "(?i)(?<=[A-Z])\\.(?=[A-Z])", options: [.caseInsensitive], with: "-")
        
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}



