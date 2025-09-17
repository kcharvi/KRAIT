// Utility to parse MDX files and extract frontmatter and code blocks

export interface Problem {
    id: string;
    name: string;
    code: string;
    description?: string;
    requirements?: string[];
}

export function parseFrontmatter(content: string): { frontmatter: any; content: string } {
    const frontmatterRegex = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/;
    const match = content.match(frontmatterRegex);
    
    if (!match) {
        return { frontmatter: {}, content };
    }
    
    const frontmatterText = match[1];
    const contentWithoutFrontmatter = match[2];
    
    // Parse YAML frontmatter (simple parsing)
    const frontmatter: any = {};
    const lines = frontmatterText.split(/\r?\n/);
    
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        
        const colonIndex = trimmed.indexOf(':');
        if (colonIndex === -1) continue;
        
        const key = trimmed.substring(0, colonIndex).trim();
        let value = trimmed.substring(colonIndex + 1).trim();
        
        // Remove quotes if present
        if ((value.startsWith('"') && value.endsWith('"')) || 
            (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
        }
        
        // Handle array values
        if (value.startsWith('[') && value.endsWith(']')) {
            const arrayContent = value.slice(1, -1);
            frontmatter[key] = arrayContent.split(',').map(item => 
                item.trim().replace(/^["']|["']$/g, '')
            );
        } else {
            frontmatter[key] = value;
        }
    }
    
    return { frontmatter, content: contentWithoutFrontmatter };
}

export function extractCodeFromMDX(content: string): string {
    // Extract code from ```python code blocks
    const codeBlockRegex = /```python\r?\n([\s\S]*?)\r?\n```/;
    const match = content.match(codeBlockRegex);
    return match ? match[1].trim() : '';
}

export function parseProblemMDX(id: string, content: string): Problem {
    const { frontmatter, content: contentWithoutFrontmatter } = parseFrontmatter(content);
    
    return {
        id,
        name: frontmatter.title || 'Untitled',
        code: extractCodeFromMDX(contentWithoutFrontmatter),
        description: frontmatter.description || '',
        requirements: frontmatter.requirements || []
    };
}
